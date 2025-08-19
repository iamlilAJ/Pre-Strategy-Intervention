import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Any
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from jaxmarl.environments.spaces import Box

# print("Using Local MPE")
class SimpleSpreadMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=3,
        num_landmarks=3,
        local_ratio=0.5,
        action_type=DISCRETE_ACT,
        train_pre_policy=False,  # Add train_pre_policy as an argument
        intrinsic_reward_ratio=0.1
    ):
        dim_c = 2  # NOTE follows code rather than docs

        # Action and observation spaces
        agents = ["agent_{}".format(i) for i in range(num_agents)]
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (4+(num_agents-1)*4+(num_landmarks*2),))
            for i in agents
        }

        colour = [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks

        # Env specific parameters
        self.local_ratio = local_ratio
        self.train_pre_policy = train_pre_policy  # Store train_pre_policy flag
        self.intrinsic_reward_ratio = intrinsic_reward_ratio

        assert (
            self.local_ratio >= 0.0 and self.local_ratio <= 1.0
        ), "local_ratio must be between 0.0 and 1.0"

        # Parameters
        rad = jnp.concatenate(
            [jnp.full((num_agents), 0.15), jnp.full((num_landmarks), 0.05)]
        )
        collide = jnp.concatenate(
            [jnp.full((num_agents), True), jnp.full((num_landmarks), False)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=dim_c,
            colour=colour,
            rad=rad,
            collide=collide,
        )

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx: int):
            """Values needed in all observations"""

            landmark_pos = (
                state.p_pos[self.num_agents :] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame

            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]
            comm = jnp.roll(
                state.c[: self.num_agents], shift=self.num_agents - aidx - 1, axis=0
            )[: self.num_agents - 1]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            comm = jnp.roll(comm, shift=aidx, axis=0)

            return landmark_pos, other_pos, comm

        landmark_pos, other_pos, comm = _common_stats(self.agent_range)

        def _obs(aidx: int):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                    comm[aidx].flatten(),
                ]
            )

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    def rewards(self, state: State) -> Dict[str, Tuple[float, float]]:
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        c = _collisions(
            self.agent_range,
            self.agent_range,
        )  # [agent, agent, collison]

        def _agent_rew(aidx: int, collisions: chex.Array):
            rew = -1 * jnp.sum(collisions[aidx])
            return rew

        def _land(land_pos: chex.Array):
            d = state.p_pos[: self.num_agents] - land_pos
            return -1 * jnp.min(jnp.linalg.norm(d, axis=1))

        # Calculate the global reward
        global_rew = jnp.sum(jax.vmap(_land)(state.p_pos[self.num_agents :]))


        # leftmost_landmark_pos = state.p_pos[self.num_agents:][0]  # The leftmost landmark (first one)
        index = jnp.argmin(state.p_pos[self.num_agents:, 0])
        leftmost_landmark_pos = state.p_pos[self.num_agents:][index]

        dist_to_leftmost_landmark = jnp.linalg.norm(state.p_pos[0] - leftmost_landmark_pos)
        additional_rew = -dist_to_leftmost_landmark  # Reward based on distance to the leftmost landmark
        rew = {
            a: (
                _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio),
                additional_rew  if i == 0 else 0.0  # Apply additional reward for agent 0
            )
            for i, a in enumerate(self.agents)
        }
        return rew

        # Add additional reward if train_pre_policy is True for the first agent
        # if self.train_pre_policy:
        #     leftmost_landmark_pos = state.p_pos[self.num_agents:][0]  # The leftmost landmark (first one)
        #     dist_to_leftmost_landmark = jnp.linalg.norm(state.p_pos[0] - leftmost_landmark_pos)
        #     additional_rew = -dist_to_leftmost_landmark  # Reward based on distance to the leftmost landmark
        #     rew = {
        #         a: (
        #             _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio),
        #             additional_rew  if i == 0 else 0.0  # Apply additional reward for agent 0
        #         )
        #         for i, a in enumerate(self.agents)
        #     }
        #     return rew

        # else:
        #     rew = {
        #         a: _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio) for i, a in enumerate(self.agents)
        #     }
        #     return rew

        # Compute the rewards for each agent



if __name__ == "__main__":
    from jaxmarl.environments.mpe import MPEVisualizer

    num_agents = 3
    key = jax.random.PRNGKey(0)

    env = SimpleSpreadMPE(num_agents)

    obs, state = env.reset(key)

    mock_action = jnp.array([[1.0, 1.0, 0.1, 0.1, 0.0]])

    actions = jnp.repeat(mock_action[None], repeats=num_agents, axis=0).squeeze()

    actions = {agent: mock_action for agent in env.agents}
    a = env.agents
    a.reverse()
    print("a", a)
    actions = {agent: mock_action for agent in a}
    print("actions", actions)

    # env.enable_render()

    state_seq = []
    print("state", state)
    print("action spaces", env.action_spaces)

    for _ in range(25):
        state_seq.append(state)
        key, key_act = jax.random.split(key)
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        obs, state, rew, dones, _ = env.step_env(key, state, actions)

    viz = MPEVisualizer(env, state_seq)
    viz.animate(None, view=True)



