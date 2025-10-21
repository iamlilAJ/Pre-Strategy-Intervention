import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
from flax.linen.initializers import orthogonal
from typing import Any, Callable

# -------------------------------------------------------------------
# 1) Preprocessor for 4 Players
# -------------------------------------------------------------------
class HanabiGraphPreprocessor4Players:
    """
    Splits a 4-player Hanabi observation (of length 1041) into meaningful sub-blocks,
    then further decomposes them into node-like structures for GNN usage.
    """
    def __init__(self):
        # Known sub-block sizes for a 4-player scenario
        self.hands_size      = 304
        self.board_size      = 70
        self.discards_size   = 50
        self.last_action_size= 57
        self.v0_belief_size  = 560

        self.obs_size = (
            self.hands_size
            + self.board_size
            + self.discards_size
            + self.last_action_size
            + self.v0_belief_size
        )
        assert self.obs_size == 1041, f"Got {self.obs_size}, expected 1041."

        # Offsets
        self.hands_start = 0
        self.hands_end   = self.hands_start + self.hands_size

        self.board_start = self.hands_end
        self.board_end   = self.board_start + self.board_size

        self.discards_start    = self.board_end
        self.discards_end      = self.discards_start + self.discards_size

        self.last_action_start = self.discards_end
        self.last_action_end   = self.last_action_start + self.last_action_size

        self.v0_belief_start = self.last_action_end
        self.v0_belief_end   = self.v0_belief_start + self.v0_belief_size

    @partial(jax.jit, static_argnums=(0,))
    def preprocess_observation(self, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Convert a single 4-player Hanabi observation (shape: (1041,))
        into a list of node feature vectors. Returns shape (num_nodes, padded_dim).
        """

        # 1) Extract Sub-Blocks
        hands_feat = observation[self.hands_start:self.hands_end]        # e.g. 304
        board_feat = observation[self.board_start:self.board_end]        # e.g. 70
        discards_feat = observation[self.discards_start:self.discards_end]   # 50
        last_action_feat = observation[self.last_action_start:self.last_action_end] # 57
        v0_belief_feat = observation[self.v0_belief_start:self.v0_belief_end]       # 560

        # ------------------------------------------------------------------
        # Example Decomposition (You can further break these down!)
        # The actual structure depends on how your code or environment
        # decided to arrange these 304, 70, etc. features.
        # We'll show a plausible breakdown for "hands_feat" or "board_feat"
        # then parse into nodes the same way you did in the 2p code.
        # ------------------------------------------------------------------

        # For 4 players, "hands_feat = 304" might be:
        #   - 3 other players × 4 cards/player × 25 features = 300
        #   - 4 leftover features? (ex: missing_card flags or something)
        other_hands = hands_feat[:300].reshape(3, 4, 25)  # shape (3 players, 4 cards, 25 feats)
        leftover_hands_info = hands_feat[300:]            # shape (4,)

        # Board: 70 could be e.g. deck(40), fireworks(25), info(3?), life(2?)
        # This is just an example; adapt to your actual structure
        deck = board_feat[:40]
        fireworks = board_feat[40:65]
        info_tokens = board_feat[65:68]  # e.g. 3 bits
        life_tokens = board_feat[68:70]  # e.g. 2 bits

        # Discards: 50 => e.g. 5 color × 10 bits each
        discards = discards_feat.reshape(5, 10)

        # Last Action: 57 => some sub-block logic
        la_acting = last_action_feat[:4]       # e.g. 4 bits for acting player index
        la_movetype = last_action_feat[4:8]    # 4 bits
        la_target   = last_action_feat[8:12]   # 4 bits
        # ... etc. for the rest ...
        la_rest     = last_action_feat[12:]

        # V0 Belief: 560 => e.g. 16 total cards × 35 feats each
        # but actual "4 player" might have 12 unseen cards, etc. The example below:
        v0_belief_cards = v0_belief_feat.reshape(16, 35)  # shape (16, 35)

        # 2) Create node-like structures
        node_list = []

        # 2.1) Hands / other players
        # Let's treat each "card" as a node
        # => we have (3 players × 4 cards) = 12 nodes
        for p in range(3):
            for c in range(4):
                node_list.append(other_hands[p, c, :])  # shape (25,)

        # leftover 4 feats can be 1 node
        node_list.append(leftover_hands_info)

        # 2.2) Board
        node_list.append(deck)
        node_list.append(fireworks)
        node_list.append(info_tokens)
        node_list.append(life_tokens)

        # 2.3) Discards => 5 nodes
        for i in range(5):
            node_list.append(discards[i])

        # 2.4) Last Action => we can break into ~6 sub-nodes or just 1 node
        node_list.append(la_acting)
        node_list.append(la_movetype)
        node_list.append(la_target)
        node_list.append(la_rest)

        # 2.5) V0 Belief => 16 card-nodes
        for i in range(16):
            node_list.append(v0_belief_cards[i])

        # 3) Pad all node vectors to a consistent dimension
        max_dim = max(n.shape[0] for n in node_list)
        padded_nodes = []
        for node in node_list:
            pad_len = max_dim - node.shape[0]
            padded = jnp.pad(node, (0, pad_len), constant_values=0)
            padded_nodes.append(padded)

        # 4) Stack => shape (num_nodes, max_dim)
        final_nodes = jnp.stack(padded_nodes, axis=0)
        return final_nodes


# -------------------------------------------------------------------
# 2) Observation Encoder for 4 Players
# -------------------------------------------------------------------
class ObservationEncoder4Players(nn.Module):
    """
    Similar to the 2-player version, but we might have more nodes, or
    the final dimension is based on the 4-player obs=1041 length.
    """
    num_nodes: int
    obs_enc_hidden_dim: int=64

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """
        Input shape: (batch_size, obs_size=1041)
        Output shape: (batch_size, num_nodes, num_nodes, 2)
        """
        x = observations

        x = nn.Dense(features=self.obs_enc_hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.num_nodes * self.num_nodes * 2, kernel_init=orthogonal(1.0))(x)
        logits = logits.reshape(x.shape[0], self.num_nodes, self.num_nodes, 2)
        return logits

# -------------------------------------------------------------------
# 3) GumbelSoftmaxAdjMatrixModel, GCNLayer, GraphMean (same as 2p)
# -------------------------------------------------------------------
class GumbelSoftmaxAdjMatrixModel(nn.Module):
    seed: int = 0
    temperature: float = 1.0

    def gumbel_softmax_sample(self, logits, rng):
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape)))
        y = logits + gumbel_noise
        return nn.softmax(y / self.temperature)

    @nn.compact
    def __call__(self, logits):
        rng = jax.random.PRNGKey(self.seed)
        gumbel_out = jax.vmap(self.gumbel_softmax_sample, in_axes=(0, None))(logits, rng)
        # shape: (batch, num_nodes, num_nodes, 2)
        # pick channel==1 => "edge exists"
        adj = gumbel_out[..., 1]
        return adj


class GCNLayer(nn.Module):
    node_feature_dim: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        """
        node_feats: (batch, num_nodes, feature_dim)
        adj_matrix: (batch, num_nodes, num_nodes)
        """
        # Basic GCN formula = Dense -> adj -> normalization
        # 1) Linear transform
        feats = nn.Dense(self.node_feature_dim, kernel_init=orthogonal(1.0))(node_feats)
        # 2) Multiply by adjacency
        # feats shape: (batch, num_nodes, feat_dim)
        # adj shape:   (batch, num_nodes, num_nodes)
        # We'll do a "batch matmul" pattern:
        # output[i] = adj[i] @ feats[i]
        out = jax.vmap(lambda A, F: A @ F)(adj_matrix, feats)
        # 3) Normalize by sum of adjacency
        denom = adj_matrix.sum(axis=-1, keepdims=True) + 1e-6
        out = out / denom
        return out


class GraphMean(nn.Module):
    @nn.compact
    def __call__(self, node_feats: jnp.ndarray) -> jnp.ndarray:
        # node_feats: (batch, num_nodes, feat_dim)
        return jnp.mean(node_feats, axis=1)  # => (batch, feat_dim)

# -------------------------------------------------------------------
# 4) End-to-End GCN for 4 Players
# -------------------------------------------------------------------
class End2EndGCN4Players(nn.Module):
    config: dict

    def setup(self):
        # Example: let's fix the num_nodes based on the logic in the preprocessor
        self.num_nodes = 1 + (3*4) + 4 + 5 + 4 + 16 # an example: leftover + (3 players × 4 cards) + board(4) + discards(5) + last_action(4) + v0(16)...

        self.preprocessor = HanabiGraphPreprocessor4Players()
        self.observation_encoder = ObservationEncoder4Players(num_nodes=self.num_nodes,
                                                              obs_enc_hidden_dim=self.config["OBS_ENC_HIDDEN_DIM"])

        self.gumbel_model = GumbelSoftmaxAdjMatrixModel(seed=self.config.get("SEED",0),
                                                        temperature=self.config.get("TEMPERATURE",1.0))
        self.gcn_layer = GCNLayer(node_feature_dim=self.config["NODE_FEATURE_DIM"])
        self.graph_mean = GraphMean()

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """
        observations shape: (batch_size, 1041)
        Return: graph embeddings shape => (batch_size, node_feature_dim)
        """

        # 1) Preprocess to node-based representation
        # returns shape: (batch_size, num_nodes, padded_dim)
        node_feats = jax.vmap(self.preprocessor.preprocess_observation)(observations)

        # 2) Encode observation -> adjacency logits
        logits = self.observation_encoder(observations)
        # shape: (batch, num_nodes, num_nodes, 2)

        # 3) Gumbel Softmax -> adjacency matrix
        adj_matrix = self.gumbel_model(logits)
        # shape: (batch, num_nodes, num_nodes)

        # 4) GCN
        feats = self.gcn_layer(node_feats, adj_matrix)
        # shape: (batch, num_nodes, node_feature_dim)

        # 5) Mean pool
        graph_emb = self.graph_mean(feats)
        # shape: (batch, node_feature_dim)

        return graph_emb

