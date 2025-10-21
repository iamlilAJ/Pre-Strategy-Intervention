import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from functools import partial

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


class PrePolicyRNN(nn.Module):
    pre_policy_hidden_dim: int
    pre_policy_output_dim: int
    init_scale: float = 1.0
    if_augment_obs: bool = True
    if_skip_connection: bool = True


    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.pre_policy_hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        if self.if_skip_connection and self.if_augment_obs:
            # FOR MPE (NUM_AGENT, BATCH_SIZE, TIMESTEP, OBS_SIZE)

            embedding = jnp.concatenate([embedding, obs[..., -1:]], axis=-1)
        pre_policy_output = nn.Dense(
            self.pre_policy_output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, pre_policy_output


class PrePolicyMLP(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    pre_policy_output_dim: int
    pre_policy_hidden_dim: int
    init_scale: float = 1.0
    pre_policy_num_layers: int = 1
    if_skip_connection: bool = True

    @nn.compact
    def __call__(self, x):

        # Assuming the intrinsic reward is the last feature in x
        intrinsic_reward = x[:, -1:]

        for l in range(self.pre_policy_num_layers):
            if self.if_skip_connection:

                x = nn.Dense(
                    self.pre_policy_hidden_dim,
                    kernel_init=orthogonal(self.init_scale),
                    bias_init=constant(0.0),
                )(x)

                x = nn.relu(x)
                # Concatenate along the feature dimension
                x = jnp.concatenate([x, intrinsic_reward], axis=-1)

        pre_policy_output = nn.Dense(
            self.pre_policy_output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)

        return pre_policy_output









