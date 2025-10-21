import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import distrax

from agent.pre_policy_module.pre_policy_network import PrePolicyMLP
from agent.gnn_module.hanabi_gnn import End2EndGCN


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class SharedMLP(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs = x

        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)
        return embedding

class Actor(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        embedding, dones, avail_actions = x

        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return pi

class Critic(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, x):
        embedding = x

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return  jnp.squeeze(critic, axis=-1)









class PrePolicyIPPO(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    num_agents: int


    def setup(self):
        self.shared_mlp = nn.vmap(SharedMLP, in_axes=0, out_axes=0,
                                  variable_axes={"params": 0},
                                  split_rngs={"params": 0})(config=self.config)

        self.gnn = nn.vmap(End2EndGCN, in_axes=0, out_axes=0,
                           variable_axes={"params": 0},
                           split_rngs={"params": 0})(config=self.config)


        self.pre_policy_network = nn.vmap(PrePolicyMLP, in_axes=0, out_axes=0,
                                          variable_axes={"params": 0},
                                          split_rngs={"params": True})(
            pre_policy_output_dim=self.config.get("PRE_POLICY_OUTPUT_DIM", 64),
            pre_policy_hidden_dim=self.config.get("PRE_POLICY_HIDDEN_DIM", 128),
        )

        self.actor = nn.vmap(Actor, in_axes=0, out_axes=0,
                             variable_axes={"params": 0},
                             split_rngs={"params": 0})(action_dim=self.action_dim,
                                                       config=self.config)

        self.critic = nn.vmap(Critic, in_axes=0, out_axes=0,
                              variable_axes={"params": 0},
                              split_rngs={"params": 0})(config=self.config)



    def __call__(self, x):
        obs, dones, avail_actions = x

        agent_embedding = self.shared_mlp(obs)

        pre_policy_embedding = self.pre_policy_network(obs)
        mask = jnp.arange(self.num_agents) < self.config["NUM_PROXY_AGENTS"]
        mask = mask.astype(jnp.float32)[:, None ,None, None]  # (num_agents, 1)
        # Non-proxy agents receive all zero pre-policy embedding
        pre_policy_embedding = pre_policy_embedding * mask  # (num_agents, pre_policy_output_dim)


        gnn_features = self.gnn(obs)

        critic_input = jnp.concatenate([agent_embedding, pre_policy_embedding, gnn_features], axis=-1)

        critic = self.critic(critic_input)

        actor_input = (jnp.concatenate([agent_embedding, gnn_features], axis=-1),
                       dones, avail_actions)

        pi = self.actor(actor_input)

        return pi, critic




class BaselineIPPO(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.agent=nn.vmap(ActorCritic, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
             self.action_dim, self.config
        )

    def __call__(self, x):
        # print(x)
        pi, critic = self.agent(x)
        return  pi, critic

