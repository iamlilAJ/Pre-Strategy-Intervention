
import flax.linen as nn

import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict, Union, Any, Tuple, Optional, List
from functools import partial
from agent.pre_policy_module.pre_policy_network import PrePolicyMLP
from agent.gnn_module.hanabi_gnn import End2EndGCN
from agent.gnn_module.hanabi_4_player_gnn import End2EndGCN4Players



class MLPNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 3
    norm_input: bool = False
    norm_type: str = "layer_norm"
    dueling: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            # x = x

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        return x

class QNetwork(nn.Module):
    action_dim: int
    dueling: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.dueling:
            adv = nn.Dense(self.action_dim)(x)
            val = nn.Dense(1)(x)
            q_vals = val + adv - jnp.mean(adv, axis=-1, keepdims=True)
        else:
            q_vals = nn.Dense(self.action_dim)(x)
        return q_vals

# No parameter sharing
class PQNAgent(nn.Module):
    action_dim: int
    num_agents: int
    config: Any
    # num_proxy_agents: int = 1
    init_scale: float = 1.0

    dueling: bool = True  # Added dueling flag

    if_augment_obs: bool = True

    def setup(self):
        # Initialize GNN

        if self.num_agents == 2:
            self.gnn = nn.vmap(End2EndGCN, in_axes=0, out_axes=0,
                           variable_axes={"params": 0},
                           split_rngs={"params": 0})(config=self.config)
        elif self.num_agents == 4:
            self.gnn = nn.vmap(End2EndGCN4Players, in_axes=0, out_axes=0,
                               variable_axes={"params": 0},
                               split_rngs={"params": 0})(config=self.config)
        else:
            raise NotImplementedError("This implementation only supports 2 or 4 agents.")

        # Initialize MLPNetwork
        self.agent_mlp = nn.vmap(MLPNetwork, in_axes=(0, None), out_axes=0,
                                 variable_axes={"params": 0, "batch_stats": 0},
                                 split_rngs={"params": True})(
            action_dim=self.action_dim,
            hidden_size=self.config.get('HIDDEN_SIZE', 512),
            num_layers=self.config.get('MLP_NUM_LAYERS', 3),
            norm_input=self.config.get('MLP_NORM_INPUT', False),
            norm_type=self.config.get('MLP_NORM_TYPE', "layer_norm"),
            dueling=False  # Dueling handled in IQLAgent
        )

        # Initialize PrePolicyNetwork
        self.pre_policy_network = nn.vmap(PrePolicyMLP, in_axes=0, out_axes=0,
                                          variable_axes={"params": 0},
                                          split_rngs={"params": True})(
            pre_policy_output_dim=self.config.get("PRE_POLICY_OUTPUT_DIM", 64),
            pre_policy_hidden_dim=self.config.get("PRE_POLICY_HIDDEN_DIM", 128),
        )

        self.q_network = nn.vmap(QNetwork, in_axes=0, out_axes=0,
                                 variable_axes={"params": 0,},
                                 split_rngs={"params": True})(action_dim=self.action_dim, dueling=self.dueling)


    def __call__(self, x, train=False):
        """
        Args:
            x: jnp.ndarray with shape (num_agents, 658)
            rng: jax.random.PRNGKey
        Returns:
            q_vals: jnp.ndarray with shape (num_agents, action_dim)
        """

        # print(f"input shape: {x.shape}")
        # print("sum for agent 0", jnp.sum(x[0, :, -1]))
        #
        # print("sum for agent 1", jnp.sum(x[1, :, -1]))
        original_obs = x[:, :, :-1] # remove last

        agent_embedding = self.agent_mlp(original_obs, train)  # (num_agents, hidden_dim)

        #
        if self.if_augment_obs:
            pre_policy_input = x
        else:
            pre_policy_input = original_obs

        pre_policy_embedding = self.pre_policy_network(pre_policy_input)  # (num_agents, pre_policy_output_dim)

        # Create a mask for pre_policy_embedding
        mask = jnp.arange(self.num_agents) < self.config["NUM_PROXY_AGENTS"]
        mask = mask.astype(jnp.float32)[:, None, None]  # (num_agents, 1)
        # Non-proxy agents receive all zero pre-policy embedding
        pre_policy_embedding = pre_policy_embedding * mask  # (num_agents, pre_policy_output_dim)

        # Process observations through MLP for GNN input

        # Generate graph embedding using GNN
        gnn_features = self.gnn(x)  # (num_agents, GNN_EMBEDDING_DIM)

        # Concatenate embeddings: [agent_embedding, gnn_features, pre_policy_embedding]
        q_value_input = jnp.concatenate([agent_embedding, gnn_features, pre_policy_embedding], axis=-1)  # (num_agents, hidden_dim + GNN_EMBEDDING_DIM + pre_policy_output_dim)

        # Dueling architecture
        q_vals =self.q_network(q_value_input)

        return q_vals



class BaselinePQNAgent(nn.Module):
    action_dim: int
    config: Any

    dueling: bool = True  # Added dueling flag

    if_augment_obs: bool = False

    def setup(self):

        self.agent_mlp = nn.vmap(MLPNetwork, in_axes=(0, None), out_axes=0,
                                 variable_axes={"params": 0, "batch_stats": 0},
                                 split_rngs={"params": True})(
            action_dim=self.action_dim,
            hidden_size=self.config.get('HIDDEN_SIZE', 512),
            num_layers=self.config.get('MLP_NUM_LAYERS', 4),
            norm_input=self.config.get('MLP_NORM_INPUT', False),
            norm_type=self.config.get('MLP_NORM_TYPE', "layer_norm"),
            dueling=self.config.get("DUELING", True)
        )

        self.q_network = nn.vmap(QNetwork, in_axes=0, out_axes=0,
                                 variable_axes={"params": 0,},
                                 split_rngs={"params": True})(action_dim=self.action_dim, dueling=self.dueling)



    def __call__(self, x, train=False):

        # if self.if_augment_obs:
        #     input = x  # remove last augmented feature.
        # else:
        #
        #     input = x[:, :, :-1]

        agent_embedding = self.agent_mlp(x, train)

        q_vals = self.q_network(agent_embedding)

        return q_vals

