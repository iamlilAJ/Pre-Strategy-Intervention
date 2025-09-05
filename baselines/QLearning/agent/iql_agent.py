
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict, Union, Any, Tuple, Optional, List
from functools import partial
from agent.pre_policy_module.pre_policy_network import PrePolicyRNN
from agent.gnn_module.mpe_gnn import End2EndGCN

class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )

class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim

    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, hidden, obs, dones):

        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.1))(obs)
        embedding = nn.relu(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        #
        # q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.1))(embedding_pre_policy)

        return hidden, embedding

class IQLAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    num_agents: int
    num_proxy_agents: int
    init_scale: float
    pre_policy_output_dim: int
    pre_policy_hidden_dim: int
    node_feature_dim: int

    if_augment_obs: bool = True

    def setup(self):

        self.gnn = nn.vmap(End2EndGCN, in_axes=0, out_axes=0,
                           variable_axes={"params": 0},
                           split_rngs={"params": 0})(node_feature_dim=self.node_feature_dim)

        self.agent_rnn=nn.vmap(AgentRNN, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
             self.hidden_dim, self.init_scale
        )

        # Initialize PrePolicyNetwork
        self.pre_policy_network = nn.vmap(PrePolicyRNN, in_axes=0, out_axes=0,
                                          variable_axes={"params": 0},
                                          split_rngs={"params": True})(
            self.pre_policy_hidden_dim, self.pre_policy_output_dim,
            self.init_scale, self.if_augment_obs,
        )

        self.mlp_gnn = nn.vmap(nn.Dense, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
            features=14, kernel_init=orthogonal(self.init_scale)
        )

        self.q_value_mlp = nn.vmap(nn.Dense, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.action_dim, kernel_init=orthogonal(self.init_scale)
        )

    def __call__(self, agent_hidden, obs, dones, pre_policy_hidden):
        '''
        All input observation processed by AgentRNN, then output embedding,
        embedding input into GNN and output graph embedding,
        Q values input: [AgentRNN output embedding, gnn graph embedding, pre_policy embedding]
        '''


        original_obs = obs[..., :-1]  # remove last augmented feature.

        agent_hidden, agent_embedding = self.agent_rnn(agent_hidden, original_obs, dones)

        if self.if_augment_obs:
            pre_policy_obs = obs
        else:
            pre_policy_obs = original_obs

        pre_policy_hidden, pre_policy_embedding = self.pre_policy_network(pre_policy_hidden, pre_policy_obs, dones)


        # Create a mask for pre_policy_embedding
        mask = jnp.arange(self.num_agents) < self.num_proxy_agents
        mask = mask[:, None, None, None]  # Reshape for broadcasting
        # Non-proxy agent receive all zero pre-policy embedding
        pre_policy_embedding = pre_policy_embedding * mask

        gnn_input = self.mlp_gnn(agent_embedding)

        gnn_features = self.gnn(gnn_input)

        q_value_input = jnp.concatenate([agent_embedding, gnn_features, pre_policy_embedding], axis=-1)
        q_vals = self.q_value_mlp(q_value_input)

        return agent_hidden, q_vals, pre_policy_hidden






class BaselineIQLAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float

    def setup(self):

        self.agent_rnn=nn.vmap(AgentRNN, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
             self.hidden_dim, self.init_scale
        )


        self.q_value_mlp = nn.vmap(nn.Dense, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.action_dim, kernel_init=orthogonal(self.init_scale)
        )

    def __call__(self, agent_hidden, obs, dones, _):
        '''
        All input observation processed by AgentRNN, then output embedding,
        embedding input into GNN and output graph embedding,
        Q values input: [AgentRNN output embedding, gnn graph embedding, pre_policy embedding]
        '''
        agent_hidden, agent_embedding = self.agent_rnn(agent_hidden, obs, dones)

        q_vals = self.q_value_mlp(agent_embedding)

        # return dummy pre-policy module hidden
        return agent_hidden, q_vals, _




