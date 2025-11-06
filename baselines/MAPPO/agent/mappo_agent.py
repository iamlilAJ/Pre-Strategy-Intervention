import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict

import distrax

from agent.pre_policy_module.pre_policy_network import PrePolicyMLP
from agent.gnn_module.hanabi_gnn import End2EndGCN

class ActorFF(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs, dones, avail_actions = x
        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return pi


class CriticFF(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(critic)

        return jnp.squeeze(critic, axis=-1)



class PrePolicyMAPPO(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    num_agents: int


    def setup(self):


        self.gnn = nn.vmap(End2EndGCN, in_axes=0, out_axes=0,
                           variable_axes={"params": 0},
                           split_rngs={"params": 0})(config=self.config)


        self.pre_policy_network = nn.vmap(PrePolicyMLP, in_axes=0, out_axes=0,
                                          variable_axes={"params": 0},
                                          split_rngs={"params": True})(
            pre_policy_output_dim=self.config.get("PRE_POLICY_OUTPUT_DIM", 64),
            pre_policy_hidden_dim=self.config.get("PRE_POLICY_HIDDEN_DIM", 128),
        )

        self.actor = nn.vmap(ActorFF, in_axes=0, out_axes=0,
                             variable_axes={"params": 0},
                             split_rngs={"params": 0})(action_dim=self.action_dim,
                                                       config=self.config)

        if self.config["SHARE_CRITIC"] is True:
            self.critic = nn.vmap(CriticFF, in_axes=0, out_axes=0,
                                  variable_axes={"params": None},
                                  split_rngs={"params": False})(config=self.config)
        else:
            self.critic = nn.vmap(CriticFF, in_axes=0, out_axes=0,
                                  variable_axes={"params": 0},
                                  split_rngs={"params": 0})(config=self.config)


    def __call__(self, x, global_obs):
        obs, dones, avail_actions = x

        pre_policy_embedding = self.pre_policy_network(obs)
        mask = jnp.arange(self.num_agents) < self.config["NUM_PROXY_AGENTS"]
        mask = mask.astype(jnp.float32)[:, None ,None, None]  # (num_agents, 1)

        pre_policy_embedding = pre_policy_embedding * mask  # (num_agents, pre_policy_output_dim)

        gnn_features = self.gnn(obs)

        critic_input = jnp.concatenate([global_obs, pre_policy_embedding, gnn_features], axis=-1)


        output_critic = self.critic(critic_input)

        if self.config["PRE_POLICY_EMD_INPUT_ACTOR"] is True:
            actor_input = (jnp.concatenate([obs, gnn_features, pre_policy_embedding], axis=-1),
                       dones, avail_actions)
        else:
            actor_input = (jnp.concatenate([obs, gnn_features], axis=-1),
                       dones, avail_actions)

        pi = self.actor(actor_input)

        return pi, output_critic

class GlobalPrePolicyMAPPO(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    num_agents: int


    def setup(self):


        self.gnn = nn.vmap(End2EndGCN, in_axes=0, out_axes=0,
                           variable_axes={"params": 0},
                           split_rngs={"params": 0})(config=self.config)


        self.pre_policy_network = nn.vmap(PrePolicyMLP, in_axes=0, out_axes=0,
                                          variable_axes={"params": 0},
                                          split_rngs={"params": True})(
            pre_policy_output_dim=self.config.get("PRE_POLICY_OUTPUT_DIM", 64),
            pre_policy_hidden_dim=self.config.get("PRE_POLICY_HIDDEN_DIM", 128),
        )

        self.actor = nn.vmap(ActorFF, in_axes=0, out_axes=0,
                             variable_axes={"params": 0},
                             split_rngs={"params": 0})(action_dim=self.action_dim,
                                                       config=self.config)

        if self.config["SHARE_CRITIC"] is True:
            self.critic = nn.vmap(CriticFF, in_axes=0, out_axes=0,
                                  variable_axes={"params": None},
                                  split_rngs={"params": False})(config=self.config)
        else:
            self.critic = nn.vmap(CriticFF, in_axes=0, out_axes=0,
                                  variable_axes={"params": 0},
                                  split_rngs={"params": 0})(config=self.config)


    def __call__(self, x, global_obs):
        obs, dones, avail_actions = x

        pre_policy_embedding = self.pre_policy_network(obs)

        gnn_features = self.gnn(obs)

        critic_input = jnp.concatenate([global_obs, pre_policy_embedding, gnn_features], axis=-1)


        output_critic = self.critic(critic_input)

        if self.config["PRE_POLICY_EMD_INPUT_ACTOR"] is True:
            actor_input = (jnp.concatenate([obs, gnn_features, pre_policy_embedding], axis=-1),
                       dones, avail_actions)
        else:
            actor_input = (jnp.concatenate([obs, gnn_features], axis=-1),
                       dones, avail_actions)

        pi = self.actor(actor_input)

        return pi, output_critic


class BaselineMAPPO(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.actor = nn.vmap(ActorFF, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.action_dim, self.config
        )

        self.critic = nn.vmap(CriticFF, in_axes=0, out_axes=0, variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.config
        )


    def __call__(self, x, global_obs):
        pi = self.actor(x)
        critic = self.critic(global_obs)

        return pi, critic


