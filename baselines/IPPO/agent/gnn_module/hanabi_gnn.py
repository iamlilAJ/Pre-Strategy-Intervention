import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
from flax.linen.initializers import orthogonal
from typing import Any, Callable

# Preprocessor Class
class HanabiGraphPreprocessor:
    def __init__(self):
        # Feature sizes given in the problem statement
        self.hands_size = 127
        self.board_size = 76
        self.discards_size = 50
        self.last_action_size = 55
        self.v0_belief_size = 350
        self.obs_size = self.hands_size + self.board_size + self.discards_size + self.last_action_size + self.v0_belief_size
        assert self.obs_size == 658, f"Total features {self.obs_size} != 658"

        # Define the splits
        self.hands_start = 0
        self.hands_end = self.hands_start + self.hands_size

        self.board_start = self.hands_end
        self.board_end = self.board_start + self.board_size

        self.discards_start = self.board_end
        self.discards_end = self.discards_start + self.discards_size

        self.last_action_start = self.discards_end
        self.last_action_end = self.last_action_start + self.last_action_size

        self.v0_belief_start = self.last_action_end
        self.v0_belief_end = self.v0_belief_start + self.v0_belief_size

    @partial(jax.jit, static_argnums=(0,))
    def preprocess_observation(self, observation):
        """
        Convert a single Hanabi observation (shape: (658,)) into multiple nodes.
        Each node is a smaller feature vector.
        """
        # Extract segments
        hands_feat = observation[self.hands_start:self.hands_end]        # 127
        board_feat = observation[self.board_start:self.board_end]        # 76
        discards_feat = observation[self.discards_start:self.discards_end] # 50
        last_action_feat = observation[self.last_action_start:self.last_action_end] # 55
        v0_belief_feat = observation[self.v0_belief_start:self.v0_belief_end] # 350

        # Decompose Hands:
        # other player hand: 5 cards × 25 features = 125
        # hands missing card: 2 features
        other_player_hand = hands_feat[:125].reshape(5, 25)  # 5 nodes
        hands_missing_card = hands_feat[125:]  # shape (2,) - 1 node

        # Decompose Board:
        # Deck: 40, Fireworks: 25, Info Tokens: 8, Life Tokens: 3
        deck = board_feat[0:40]
        fireworks = board_feat[40:65]
        info_tokens = board_feat[65:73]
        life_tokens = board_feat[73:76]

        # Discards: 5 colors × 10 bits each = 50
        discards = discards_feat.reshape(5, 10)  # 5 nodes

        # Last Action:
        # Acting player index: 2
        # MoveType: 4
        # Target player index: 2
        # Color revealed: 5
        # Rank revealed: 5
        # Reveal outcome: 5
        # Position played/discarded: 5
        # Card played/discarded: 25
        # Card played scored: 1
        # Card played added info token: 1
        la_acting_player = last_action_feat[0:2]
        la_movetype = last_action_feat[2:6]
        la_target_player = last_action_feat[6:8]
        la_color_revealed = last_action_feat[8:13]
        la_rank_revealed = last_action_feat[13:18]
        la_reveal_outcome = last_action_feat[18:23]
        la_position = last_action_feat[23:28]
        la_card_played_discarded = last_action_feat[28:53]
        la_card_played_scored = last_action_feat[53:54]
        la_card_played_info = last_action_feat[54:55]

        # V0 Belief: 350 = 10 cards × (25 + 5 + 5) = 10 cards × 35 features
        # first 250 bits: possible card (25 bits per card × 10 cards)
        # next 50 bits: color hinted (5 bits per card × 10 cards)
        # last 50 bits: rank hinted (5 bits per card × 10 cards)
        possible_card = v0_belief_feat[:250].reshape(10, 25)
        color_hinted = v0_belief_feat[250:300].reshape(10, 5)
        rank_hinted = v0_belief_feat[300:350].reshape(10, 5)

        # Combine V0 belief per card: each card node = 25 + 5 + 5 = 35 features
        v0_belief_nodes = jnp.concatenate([possible_card, color_hinted, rank_hinted], axis=-1)  # shape (10, 35)

        # Now stack all nodes:
        # Hands: 5 card-nodes of shape (25,), 1 missing-card node of shape (2,)
        # Board: deck(40), fireworks(25), info_tokens(8), life_tokens(3)
        # Discards: 5 nodes (each 10 features)
        # Last action: 10 nodes
        # V0 belief: 10 nodes

        # We'll store them in a list first
        node_list = []
        # Add hand nodes
        for i in range(5):
            node_list.append(other_player_hand[i])  # (25,)
        node_list.append(hands_missing_card)  # (2,)

        # Add board nodes
        node_list.append(deck)
        node_list.append(fireworks)
        node_list.append(info_tokens)
        node_list.append(life_tokens)

        # Add discard nodes (5 colors)
        for i in range(5):
            node_list.append(discards[i])

        # Add last action nodes
        la_nodes = [la_acting_player, la_movetype, la_target_player, la_color_revealed,
                    la_rank_revealed, la_reveal_outcome, la_position, la_card_played_discarded,
                    la_card_played_scored, la_card_played_info]
        node_list.extend(la_nodes)

        # Add v0 belief nodes (10 cards)
        for i in range(10):
            node_list.append(v0_belief_nodes[i])

        # Pad all nodes to the same dimension if desired, or just return a ragged list.
        # For a GNN, usually, we want a uniform dimension. We can find the max dimension:
        max_dim = max(node.shape[0] for node in node_list)
        # Pad each node to max_dim
        padded_nodes = []
        for node in node_list:
            pad_len = max_dim - node.shape[0]
            padded_node = jnp.pad(node, (0, pad_len))
            padded_nodes.append(padded_node)

        # Stack into (num_nodes, max_dim)
        final_nodes = jnp.stack(padded_nodes, axis=0)
        return final_nodes

# Observation Encoder Class
class ObservationEncoder(nn.Module):
    """Encodes observations into logits for adjacency."""
    num_nodes: int
    num_layers: int = 1
    obs_enc_hidden_dim: int = 64  # increased dimension for complexity

    @nn.compact
    def __call__(self, observations):
        x = observations
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.obs_enc_hidden_dim)(x)
            x = nn.relu(x)
        logits = nn.Dense(features=self.num_nodes * self.num_nodes * 2, kernel_init=orthogonal(1.0))(x)
        logits = logits.reshape(observations.shape[0], self.num_nodes, self.num_nodes, 2)
        return logits

# Gumbel Softmax Adjacency Matrix Model
class GumbelSoftmaxAdjMatrixModel(nn.Module):
    seed: int = 1
    temperature: float = 1.0

    def gumbel_softmax_sample(self, logits, rng):
        # Sample Gumbel noise
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape)))
        y = logits + gumbel_noise
        return nn.softmax(y / self.temperature)

    @nn.compact
    def __call__(self, logits):
        # Get RNG
        rng = jax.random.PRNGKey(self.seed)
        # vmap over batch dimension
        gumbel_softmax_output = (jax.vmap(lambda l, r: self.gumbel_softmax_sample(l, r))
                                 (logits, jax.random.split(rng, logits.shape[0])))
        # Extract "edge" probability
        soft_adj_matrix = gumbel_softmax_output[..., 1]
        return soft_adj_matrix

# GCN Layer Class
class GCNLayer(nn.Module):
    node_feature_dim: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        num_neighbours = adj_matrix.sum(axis=-1, keepdims=True)
        node_feats = nn.Dense(features=self.node_feature_dim, kernel_init=orthogonal(1.0))(node_feats)
        node_feats = jax.vmap(lambda adj, feat: jnp.dot(adj, feat))(adj_matrix, node_feats)
        node_feats = node_feats / (num_neighbours + 1e-6)
        return node_feats

# Graph Mean Pooling Class
class GraphMean(nn.Module):
    @nn.compact
    def __call__(self, node_feats):
        return jnp.mean(node_feats, axis=-2)

# End-to-End GCN Module
class End2EndGCN(nn.Module):
    config: Any
    obs_enc_hidden_dim: int = 64
    temperature: float = 1.0


    def setup(self):
        # For Hanabi: Based on the preprocessor, we have 35 nodes
        self.num_nodes = 35
        self.preprocessor = HanabiGraphPreprocessor()
        self.observation_encoder = ObservationEncoder(num_nodes=self.num_nodes,
                                                      obs_enc_hidden_dim=self.config["OBS_ENC_HIDDEN_DIM"],)
        self.gumbel_softmax_model = GumbelSoftmaxAdjMatrixModel(seed=self.config['SEED'],
                                                                temperature=self.config["TEMPERATURE"])
        self.gcn_layer = GCNLayer(node_feature_dim=self.config["NODE_FEATURE_DIM"],)
        self.graph_mean = GraphMean()

    def __call__(self, observations):
        """
        observations: jnp.ndarray with shape (time_step, batch_size, 658)
        rng: jax.random.PRNGKey
        """
        # observations shape: (time_step, batch_size, 658)
        time_step, batch_size, obs_size = observations.shape

        observations = observations.reshape(time_step * batch_size, obs_size)

        # Preprocess into node features
        node_feats = jax.vmap(self.preprocessor.preprocess_observation)(observations)
        # node_feats shape: (time_step * batch_size, num_nodes, max_dim)

        # Encode observations to produce adjacency logits
        logits = self.observation_encoder(observations)
        # logits shape: (time_step * batch_size, num_nodes, num_nodes, 2)

        # Generate soft adjacency matrix
        adj_matrix = self.gumbel_softmax_model(logits)
        # adj_matrix shape: (time_step * batch_size, num_nodes, num_nodes)

        # GCN Layer
        node_feats = self.gcn_layer(node_feats, adj_matrix)
        # node_feats: (time_step * batch_size, num_nodes, c_out)

        # Mean pooling
        graph_embedding = self.graph_mean(node_feats)
        graph_embedding = graph_embedding.reshape(time_step, batch_size, -1)

        return graph_embedding

