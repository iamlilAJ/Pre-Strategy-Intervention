import sys

from PIL.features import features

print("Python search paths:")
for path in sys.path:
    print(path)

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

class RewardRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim

    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, hidden, obs, dones):

        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.1))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        # output reward
        rew = nn.Dense(1, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.1))(embedding)

        rew = nn.tanh(rew)

        return hidden, rew


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




class LearnRewIQLAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    num_agents: int
    num_proxy_agents: int
    init_scale: float
    pre_policy_output_dim: int
    pre_policy_hidden_dim: int
    node_feature_dim: int

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
            self.pre_policy_output_dim, self.pre_policy_hidden_dim
        )

        self.mlp_gnn = nn.vmap(nn.Dense, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
            features=14, kernel_init=orthogonal(self.init_scale)
        )

        self.q_value_mlp = nn.vmap(nn.Dense, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.action_dim, kernel_init=orthogonal(self.init_scale)
        )

        # self.rew_network = nn.Sequential([nn.Dense(features=1),
        #                                   nn.tanh,
        #                                   ])
        self.rew_rnn = RewardRNN(hidden_dim=self.hidden_dim, init_scale=self.init_scale)

    def __call__(self, agent_hidden, obs, dones, pre_policy_hidden, rew_hidden):
        '''
        All input observation processed by AgentRNN, then output embedding,
        embedding input into GNN and output graph embedding,
        Q values input: [AgentRNN output embedding, gnn graph embedding, pre_policy embedding]
        '''
        agent_hidden, agent_embedding = self.agent_rnn(agent_hidden, obs, dones)

        pre_policy_hidden, pre_policy_embedding = self.pre_policy_network(pre_policy_hidden, obs, dones)

        # Create a mask for pre_policy_embedding
        mask = jnp.arange(self.num_agents) < self.num_proxy_agents
        mask = mask[:, None, None, None]  # Reshape for broadcasting
        # Non-proxy agent receive all zero pre-policy embedding
        pre_policy_embedding = pre_policy_embedding * mask

        gnn_input = self.mlp_gnn(agent_embedding)

        gnn_features = self.gnn(gnn_input)

        q_value_input = jnp.concatenate([agent_embedding, gnn_features, pre_policy_embedding], axis=-1)
        q_vals = self.q_value_mlp(q_value_input)

        # additional reward
        rew_hidden, learn_rew = self.rew_rnn(rew_hidden, agent_embedding[0, :, :, :], dones[0, :, :])

        learn_rew = learn_rew.squeeze() - 10.
        # print("learn rew shape ", learn_rew.shape)

        return agent_hidden, q_vals, pre_policy_hidden, learn_rew, rew_hidden


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



def test_iql_agent():
    """
    Test the IQLAgent with actual implementations and verify output shapes and masking.
    """
    # Define test parameters
    num_agents = 4
    timesteps = 5
    batch_size = 2
    obs_dim = 14  # Updated from 20 to 14
    action_dim = 3
    hidden_dim = 16
    num_proxy_agents = 2
    pre_policy_output_dim = 2
    pre_policy_hidden_dim = 12
    gnn_embedding_dim = 14
    init_scale = 1.0

    config = {'GNN_EMBEDDING_DIM': gnn_embedding_dim}

    # Initialize the agent
    agent = IQLAgent(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_agents=num_agents,
        num_proxy_agents=num_proxy_agents,
        init_scale=init_scale,
        pre_policy_output_dim=pre_policy_output_dim,
        pre_policy_hidden_dim=pre_policy_hidden_dim,
        config=config
    )

    # Initialize parameters with a random key
    rng = jax.random.PRNGKey(0)

    # Define the input shapes
    # agent_hidden: [num_agents, batch_size, hidden_dim]
    agent_hidden = jnp.zeros((num_agents, batch_size, hidden_dim))
    # obs: [num_agents, timesteps, batch_size, obs_dim=14]
    obs = jnp.ones((num_agents, timesteps, batch_size, obs_dim))
    # dones: [num_agents, timesteps, batch_size]
    dones = jnp.zeros((num_agents, timesteps, batch_size))
    # pre_policy_hidden: [num_agents, batch_size, pre_policy_hidden_dim]
    pre_policy_hidden = jnp.zeros((num_agents, batch_size, pre_policy_hidden_dim))

    # Initialize the parameters using dummy inputs
    variables = agent.init(rng, agent_hidden, obs, dones, pre_policy_hidden)
    # Apply the agent
    agent_hidden_out, q_vals, pre_policy_hidden_out = agent.apply(
        variables, agent_hidden, obs, dones, pre_policy_hidden
    )

    # Assertions to verify output shapes
    assert agent_hidden_out.shape == (num_agents, batch_size, hidden_dim), (
        f"agent_hidden_out shape mismatch: expected {(num_agents, batch_size, hidden_dim)}, "
        f"got {agent_hidden_out.shape}"
    )
    # assert q_vals.shape == (num_agents, batch_size, action_dim), (
    #     f"q_vals shape mismatch: expected {(num_agents, batch_size, action_dim)}, "
    #     f"got {q_vals.shape}"
    # )
    assert pre_policy_hidden_out.shape == (num_agents, batch_size, pre_policy_hidden_dim), (
        f"pre_policy_hidden_out shape mismatch: expected {(num_agents, batch_size, pre_policy_hidden_dim)}, "
        f"got {pre_policy_hidden_out.shape}"
    )

    # To verify the masking, create a subclass that returns pre_policy_embedding
    class TestableIQLAgent(IQLAgent):
        def __call__(self, agent_hidden, obs, dones, pre_policy_hidden):
            agent_hidden, agent_embedding = jax.vmap(self.agent_rnn, in_axes=0)(
                agent_hidden, obs, dones
            )

            pre_policy_hidden, pre_policy_embedding = jax.vmap(self.pre_policy_network, in_axes=0)(
                pre_policy_hidden, obs, dones
            )

            # Create a mask for pre_policy_embedding
            mask = jnp.arange(self.num_agents) < self.num_proxy_agents
            mask = mask[:, None, None, None]  # Reshape for broadcasting

            # Non-proxy agents receive all-zero pre-policy embeddings
            pre_policy_embedding = pre_policy_embedding * mask
            print(pre_policy_embedding)
            # Apply MLP to agent embeddings before GNN
            gnn_input = jax.vmap(self.mlp_gnn, in_axes=0)(agent_embedding)
            print(f'gnn input: {gnn_input.shape}')  # Expected: (num_agents, timesteps, batch_size, 14)

            # Apply GNN to the processed agent embeddings
            gnn_features = jax.vmap(self.gnn, in_axes=0)(gnn_input)
            print(f'gnn features shape: {gnn_features.shape}')  # Expected: (num_agents, timesteps, batch_size, out_embedding_dim)

            # Concatenate embeddings for Q-value computation
            q_value_input = jnp.concatenate(
                [agent_embedding, gnn_features, pre_policy_embedding], axis=-1
            )
            q_vals = jax.vmap(self.q_value_mlp, in_axes=0)(q_value_input)

            return agent_hidden, q_vals, pre_policy_hidden, pre_policy_embedding

    # Initialize the testable agent
    testable_agent = TestableIQLAgent(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_agents=num_agents,
        num_proxy_agents=num_proxy_agents,
        init_scale=init_scale,
        pre_policy_output_dim=pre_policy_output_dim,
        pre_policy_hidden_dim=pre_policy_hidden_dim,
        config=config
    )

    # Initialize parameters for the testable agent
    testable_variables = testable_agent.init(rng, agent_hidden, obs, dones, pre_policy_hidden)

    # Apply the testable agent
    _, _, _, pre_policy_embedding_out = testable_agent.apply(
        testable_variables, agent_hidden, obs, dones, pre_policy_hidden
    )

    # Create the expected mask
    expected_mask = jnp.arange(num_agents) < num_proxy_agents
    expected_mask = expected_mask[:, None, None, None]  # Shape: [num_agents, 1, 1, 1]

    # Check that pre_policy_embedding is zero for non-proxy agents
    non_proxy_agents = ~ (jnp.arange(num_agents) < num_proxy_agents)
    non_proxy_agents = non_proxy_agents[:, None, None, None]  # Reshape for broadcasting

    # Extract pre_policy_embedding for non-proxy agents
    masked_pre_policy = pre_policy_embedding_out * non_proxy_agents

    # Assert that the masked pre_policy_embedding is all zeros
    assert jnp.all(masked_pre_policy == 0), "pre_policy_embedding is not zero for non-proxy agents"

    print("All tests passed successfully.")


if __name__ == "__main__":
    test_iql_agent()








