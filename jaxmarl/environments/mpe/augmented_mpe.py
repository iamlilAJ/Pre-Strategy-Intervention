import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Any
from functools import partial
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.mpe.default_params import *
from jaxmarl.environments.spaces import Box, Discrete
from flax import struct


@struct.dataclass
class State:
    """Basic MPE State"""

    p_pos: chex.Array  # [num_entities, [x, y]]
    p_vel: chex.Array  # [n, [x, y]]
    c: chex.Array  # communication state [num_agents, [dim_c]]
    done: chex.Array  # bool [num_agents, ]
    step: int  # current step

    # last_intrinsic_reward: float
    last_intrinsic_reward: chex.Array # Vector of shape (num_agents,), storing each agent's last-step intrinsic reward

    goal: int = None  # index of target landmark, used in: SimpleSpeakerListenerMPE, SimpleReferenceMPE, SimplePushMPE, SimpleAdversaryMPE




class SimpleMPE_v2(MultiAgentEnv):
    def __init__(
        self,
        num_agents=1,
        action_type=DISCRETE_ACT,
        agents=None,
        num_landmarks=1,
        landmarks=None,
        action_spaces=None,
        observation_spaces=None,
        colour=None,
        dim_c=0,
        dim_p=2,
        max_steps=MAX_STEPS,
        dt=DT,
        train_pre_policy=True,
        all_agents_intrinsic=False,
        zero_index=True,
        **kwargs,
    ):
        # Agent and entity constants
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.agent_range = jnp.arange(num_agents)
        self.entity_range = jnp.arange(self.num_entities)
        self.train_pre_policy = train_pre_policy
        self.all_agents_intrinsic = all_agents_intrinsic
        self.zero_index = zero_index

        # Setting, and sense checking, entity names and agent action spaces
        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert (
                len(agents) == num_agents
            ), f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}
        self.classes = self.create_agent_classes()

        if landmarks is None:
            self.landmarks = [f"landmark {i}" for i in range(num_landmarks)]
        else:
            assert (
                len(landmarks) == num_landmarks
            ), f"Number of landmarks {len(landmarks)} does not match number of landmarks {num_landmarks}"
            self.landmarks = landmarks
        self.l_to_i = {l: i + self.num_agents for i, l in enumerate(self.landmarks)}

        if action_spaces is None:
            if action_type == DISCRETE_ACT:
                self.action_spaces = {i: Discrete(5) for i in self.agents}
            elif action_type == CONTINUOUS_ACT:
                self.action_spaces = {i: Box(0.0, 1.0, (5,)) for i in self.agents}
        else:
            assert (
                len(action_spaces.keys()) == num_agents
            ), f"Number of action spaces {len(action_spaces.keys())} does not match number of agents {num_agents}"
            self.action_spaces = action_spaces

        if observation_spaces is None:
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (4,)) for i in self.agents
            }
        else:
            assert (
                len(observation_spaces.keys()) == num_agents
            ), f"Number of observation spaces {len(observation_spaces.keys())} does not match number of agents {num_agents}"
            self.observation_spaces = observation_spaces

        self.colour = (
            colour
            if colour is not None
            else [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks
        )

        # Action type
        if action_type == DISCRETE_ACT:
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_decoder = self._decode_continuous_action
        else:
            raise NotImplementedError(f"Action type: {action_type} is not supported")

        # World dimensions
        self.dim_c = dim_c  # communication channel dimensionality
        self.dim_p = dim_p  # position dimensionality

        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        if "rad" in kwargs:
            self.rad = kwargs["rad"]
            assert (
                len(self.rad) == self.num_entities
            ), f"Rad array length {len(self.rad)} does not match number of entities {self.num_entities}"
            assert jnp.all(self.rad > 0), f"Rad array must be positive, got {self.rad}"
        else:
            self.rad = jnp.concatenate(
                [jnp.full((self.num_agents), 0.15), jnp.full((self.num_landmarks), 0.2)]
            )

        if "moveable" in kwargs:
            self.moveable = kwargs["moveable"]
            assert (
                len(self.moveable) == self.num_entities
            ), f"Moveable array length {len(self.moveable)} does not match number of entities {self.num_entities}"
            assert (
                self.moveable.dtype == bool
            ), f"Moveable array must be boolean, got {self.moveable}"
        else:
            self.moveable = jnp.concatenate(
                [
                    jnp.full((self.num_agents), True),
                    jnp.full((self.num_landmarks), False),
                ]
            )

        if "silent" in kwargs:
            self.silent = kwargs["silent"]
            assert (
                len(self.silent) == self.num_agents
            ), f"Silent array length {len(self.silent)} does not match number of agents {self.num_agents}"
        else:
            self.silent = jnp.full((self.num_agents), 1)

        if "collide" in kwargs:
            self.collide = kwargs["collide"]
            assert (
                len(self.collide) == self.num_entities
            ), f"Collide array length {len(self.collide)} does not match number of entities {self.num_entities}"
        else:
            self.collide = jnp.full((self.num_entities), False)

        if "mass" in kwargs:
            self.mass = kwargs["mass"]
            assert (
                len(self.mass) == self.num_entities
            ), f"Mass array length {len(self.mass)} does not match number of entities {self.num_entities}"
            assert jnp.all(
                self.mass > 0
            ), f"Mass array must be positive, got {self.mass}"
        else:
            self.mass = jnp.full((self.num_entities), 1.0)

        if "accel" in kwargs:
            self.accel = jnp.array(kwargs["accel"])
            assert (
                len(self.accel) == self.num_agents
            ), f"Accel array length {len(self.accel)} does not match number of agents {self.num_agents}"
            assert jnp.all(
                self.accel > 0
            ), f"Accel array must be positive, got {self.accel}"
        else:
            self.accel = jnp.full((self.num_agents), 5.0)

        if "max_speed" in kwargs:
            self.max_speed = kwargs["max_speed"]
            assert (
                len(self.max_speed) == self.num_entities
            ), f"Max speed array length {len(self.max_speed)} does not match number of entities {self.num_entities}"
        else:
            self.max_speed = jnp.concatenate(
                [jnp.full((self.num_agents), -1), jnp.full((self.num_landmarks), 0.0)]
            )

        if "u_noise" in kwargs:
            self.u_noise = kwargs["u_noise"]
            assert (
                len(self.u_noise) == self.num_agents
            ), f"U noise array length {len(self.u_noise)} does not match number of agents {self.num_agents}"
        else:
            self.u_noise = jnp.full((self.num_agents), 0)

        if "c_noise" in kwargs:
            self.c_noise = kwargs["c_noise"]
            assert (
                len(self.c_noise) == self.num_agents
            ), f"C noise array length {len(self.c_noise)} does not match number of agents {self.num_agents}"
        else:
            self.c_noise = jnp.full((self.num_agents), 0)

        if "damping" in kwargs:
            self.damping = kwargs["damping"]
            assert (
                self.damping >= 0
            ), f"Damping must be non-negative, got {self.damping}"
        else:
            self.damping = DAMPING

        if "contact_force" in kwargs:
            self.contact_force = kwargs["contact_force"]
        else:
            self.contact_force = CONTACT_FORCE

        if "contact_margin" in kwargs:
            self.contact_margin = kwargs["contact_margin"]
        else:
            self.contact_margin = CONTACT_MARGIN

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        u, c = self.set_actions(actions)
        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)
        done = jnp.full((self.num_agents), state.step >= self.max_steps)

        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        obs = self.get_obs(state)

        # After getting observation from the state, replace the intrinsic reward in the state.
        # state = state.replace(last_intrinsic_reward=reward['agent_0'][1])

        all_intrinsics = jnp.array([reward[a][1] for a in self.agents])
        state = state.replace(last_intrinsic_reward = all_intrinsics)

        print(state.last_intrinsic_reward)

        info = {}

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info


    def _get_leftmost_landmark(self, p_pos: jnp.ndarray):
        index = jnp.argmin(p_pos[self.num_agents:, 0])
        print("leftmost ", index)
        leftmost_landmark_pos = p_pos[self.num_agents:][index]
        return leftmost_landmark_pos

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Initialise with random positions"""

        key_a, key_l = jax.random.split(key)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(
                    key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                ),
                jax.random.uniform(
                    key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
                ),
            ]
        )
        #
        # # Initialize agent positions: shape (num_agents, 2)
        # agents_pos = jax.random.uniform(
        #     key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
        # )
        #
        # # Initialize landmark positions: shape (num_landmarks, 2)
        landmarks_pos = jax.random.uniform(
            key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
        )
        #
        # # Reorder landmarks: place the leftmost landmark first
        # sorted_landmarks_pos = jax.lax.sort(landmarks_pos, dimension=0)
        #
        # # Concatenate agents' positions with sorted landmarks' positions
        # p_pos = jnp.concatenate([agents_pos, sorted_landmarks_pos], axis=0)

        zero_landmark_pos = p_pos[self.num_agents:][0]  # index 0 among the landmarks

        if self.all_agents_intrinsic:
            @partial(jax.vmap, in_axes=(0,))
            def _intrinsic_init(aidx: int) -> float:
                dist = jnp.linalg.norm(p_pos[aidx] - zero_landmark_pos)
                return -dist

            init_intrinsic = _intrinsic_init(self.agent_range)  # shape (num_agents,)
        elif self.train_pre_policy:
            # only agent_0
            dist0 = jnp.linalg.norm(p_pos[0] - zero_landmark_pos)
            init_intrinsic = jnp.zeros((self.num_agents,))
            init_intrinsic = init_intrinsic.at[0].set(-dist0)
        else:
            init_intrinsic = jnp.zeros((self.num_agents,))

        # if self.train_pre_policy:
        #     leftmost_landmark_pos = p_pos[self.num_agents:][0]  # The leftmost landmark (first one)
        #     # index = jnp.argmin(p_pos[self.num_agents:, 0])
        #     # leftmost_landmark_pos = p_pos[self.num_agents:][index]
        #
        #     # leftmost_landmark_pos = self._get_leftmost_landmark(p_pos)
        #     dist_to_leftmost_landmark = jnp.linalg.norm(p_pos[0] - leftmost_landmark_pos)
        #     intrinsic_rew = -dist_to_leftmost_landmark  # Reward based on distance to the leftmost landmark
        # else:
        #     intrinsic_rew = 0.


        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            last_intrinsic_reward=init_intrinsic,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return dictionary of agent observations"""

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            """Return observation for agent i."""
            landmark_rel_pos = state.p_pos[self.num_agents :] - state.p_pos[aidx]

            return jnp.concatenate(
                [state.p_vel[aidx].flatten(), landmark_rel_pos.flatten()]
            )

        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def rewards(self, state: State) -> Dict[str, float]:
        """Assign rewards for all agents"""

        @partial(jax.vmap, in_axes=[0, None])
        def _reward(aidx: int, state: State):
            return -1 * jnp.sum(
                jnp.square(state.p_pos[aidx] - state.p_pos[self.num_agents :])
            )

        r = _reward(self.agent_range, state)
        return {agent: r[i] for i, agent in enumerate(self.agents)}

    def set_actions(self, actions: Dict):
        """Extract u and c actions for all agents from actions Dict."""

        actions = jnp.array([actions[i] for i in self.agents]).reshape(
            (self.num_agents, -1)
        )

        return self.action_decoder(self.agent_range, actions)

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_continuous_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.array([action[2] - action[1], action[4] - action[3]])
        u = u * self.accel[a_idx] * self.moveable[a_idx]
        c = action[5:]
        return u, c

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_discrete_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.zeros((self.dim_p,))
        idx = jax.lax.select(action <= 2, 0, 1)
        u_val = jax.lax.select(action % 2 == 0, 1.0, -1.0) * (action != 0)
        u = u.at[idx].set(u_val)
        u = u * self.accel[a_idx] * self.moveable[a_idx]
        return u, jnp.zeros((self.dim_c,))

    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array):
        p_force = jnp.zeros((self.num_agents, 2))

        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        p_force = self._apply_action_force(
            key_noise, p_force, u, self.u_noise, self.moveable[: self.num_agents]
        )
        # jax.debug.print('jax p_force post agent {p_force}', p_force=p_force)

        # apply environment forces
        p_force = jnp.concatenate([p_force, jnp.zeros((self.num_landmarks, 2))])
        p_force = self._apply_environment_force(p_force, state)
        # print('p_force post apply env force', p_force)
        # jax.debug.print('jax p_force final: {p_force}', p_force=p_force)

        # integrate physical state
        p_pos, p_vel = self._integrate_state(
            p_force, state.p_pos, state.p_vel, self.mass, self.moveable, self.max_speed
        )

        return p_pos, p_vel

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def _apply_comm_action(
        self, key: chex.PRNGKey, c: chex.Array, c_noise: int, silent: int
    ) -> chex.Array:
        silence = jnp.zeros(c.shape)
        noise = jax.random.normal(key, shape=c.shape) * c_noise
        return jax.lax.select(silent, silence, c + noise)

    # gather agent action forces
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def _apply_action_force(
        self,
        key: chex.PRNGKey,
        p_force: chex.Array,
        u: chex.Array,
        u_noise: int,
        moveable: bool,
    ):
        noise = jax.random.normal(key, shape=u.shape) * u_noise
        return jax.lax.select(moveable, u + noise, p_force)

    def _apply_environment_force(self, p_force_all: chex.Array, state: State):
        """gather physical forces acting on entities"""

        @partial(jax.vmap, in_axes=[0])
        def __env_force_outer(idx: int):
            @partial(jax.vmap, in_axes=[None, 0])
            def __env_force_inner(idx_a: int, idx_b: int):
                l = idx_b <= idx_a
                l_a = jnp.zeros((2, 2))

                collision_force = self._get_collision_force(idx_a, idx_b, state)

                xx = jax.lax.select(l, l_a, collision_force)
                # jax.debug.print('{a} {b} {f}', a=idx_a, b=idx_b, f=xx)
                return xx

            p_force_t = __env_force_inner(idx, self.entity_range)

            p_force_a = jnp.sum(p_force_t[:, 0], axis=0)  # ego force from other agents
            p_force_o = p_force_t[:, 1]
            p_force_o = p_force_o.at[idx].set(p_force_a)

            return p_force_o

        p_forces = __env_force_outer(self.entity_range)
        p_forces = jnp.sum(p_forces, axis=0)

        return p_forces + p_force_all

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0, 0])
    def _integrate_state(self, p_force, p_pos, p_vel, mass, moveable, max_speed):
        """integrate physical state"""

        p_pos += p_vel * self.dt
        p_vel = p_vel * (1 - self.damping)

        p_vel += (p_force / mass) * self.dt * moveable

        speed = jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1]))
        over_max = (
            p_vel / jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1])) * max_speed
        )

        p_vel = jax.lax.select((speed > max_speed) & (max_speed >= 0), over_max, p_vel)

        return p_pos, p_vel

    # get collision forces for any contact between two entities BUG
    def _get_collision_force(self, idx_a: int, idx_b: int, state: State):
        dist_min = self.rad[idx_a] + self.rad[idx_b]
        delta_pos = state.p_pos[idx_a] - state.p_pos[idx_b]

        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))

        # softmax penetration
        k = self.contact_margin
        penetration = jnp.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force * self.moveable[idx_a]
        force_b = -force * self.moveable[idx_b]
        force = jnp.array([force_a, force_b])

        c = (~self.collide[idx_a]) | (~self.collide[idx_b]) | (idx_a == idx_b)
        c_force = jnp.zeros((2, 2))
        return jax.lax.select(c, c_force, force)

    def create_agent_classes(self):
        if hasattr(self, "leader"):
            return {
                "leadadversary": self.leader,
                "adversaries": self.adversaries,
                "agents": self.good_agents,
            }
        elif hasattr(self, "adversaries"):
            return {
                "adversaries": self.adversaries,
                "agents": self.good_agents,
            }
        else:
            return {
                "agents": self.agents,
            }

    def agent_classes(self) -> Dict[str, list]:
        return self.classes

    ### === UTILITIES === ###
    def is_collision(self, a: int, b: int, state: State):
        """check if two entities are colliding"""
        dist_min = self.rad[a] + self.rad[b]
        delta_pos = state.p_pos[a] - state.p_pos[b]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return (dist < dist_min) & (self.collide[a] & self.collide[b]) & (a != b)

    @partial(jax.vmap, in_axes=(None, 0))
    def map_bounds_reward(self, x: float):
        """vmap over x, y coodinates"""
        w = x < 0.9
        m = x < 1.0
        mr = (x - 0.9) * 10
        br = jnp.min(jnp.array([jnp.exp(2 * x - 2), 10]))
        return jax.lax.select(m, mr, br) * ~w




class AugmentedMPE(SimpleMPE_v2):
    def __init__(
        self,
        num_agents=3,
        num_landmarks=3,
        local_ratio=0.5,
        action_type=DISCRETE_ACT,
        train_pre_policy=False,  # Add train_pre_policy as an argument
        intrinsic_reward_ratio=0.2,
        if_augment_obs=False,
        all_agents_intrinsic=False,
        zero_index=True,
        **kwargs,
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
        self.if_augment_obs = if_augment_obs
        self.all_agents_intrinsic = all_agents_intrinsic
        self.zero_index = zero_index

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
            train_pre_policy=train_pre_policy,
            all_agents_intrinsic=all_agents_intrinsic,
            zero_index=zero_index,
            **kwargs,
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

        augmented_obs = {}
        for i, a in enumerate(self.agents):
            state_obs = _obs(i)

            if self.all_agents_intrinsic and self.if_augment_obs:
                augment_feat = jnp.array([state.last_intrinsic_reward[i]])

            elif self.train_pre_policy and self.if_augment_obs and i==0:
                augment_feat = jnp.array([state.last_intrinsic_reward[i]])
            else:
                augment_feat = jnp.array([0.])

            updated_obs = jnp.concatenate([state_obs, augment_feat], axis=0)
            augmented_obs[a] = updated_obs

        return augmented_obs
        #
        #
        # obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        # return obs

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

        def _agent_extrinsic(i: int):
            return _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio)

        if self.all_agents_intrinsic and self.zero_index:
            # 1) All agents each get an intrinsic from landmark[0].
            #    Their final reward is (extrinsic, that_intrinsic).

            zero_landmark_pos = state.p_pos[self.num_agents:][0]

            @partial(jax.vmap, in_axes=(0,))
            def _intrinsic(aidx: int) -> float:
                dist_ = jnp.linalg.norm(state.p_pos[aidx] - zero_landmark_pos)
                return -dist_

            intrinsic_vals = _intrinsic(self.agent_range)

            # So each agent gets a 2-tuple: (extrinsic, intrinsic_vals[i]).
            # Possibly ignoring our "additional_rew" for agent0?
            # That depends on whether you want them both or not.
            # If you'd like both, you can combine them. For example:
            #   agent0's "intrinsic" could be 'intrinsic_vals[0] + additional_rew'.
            # The code below just shows separate extrinsic + that intrinsic.
            # You might integrate them carefully if needed.


            rew = {
                a: (
                    _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio),
                    intrinsic_vals[i]   # Apply additional reward for agent 0
                )
                for i, a in enumerate(self.agents)
            }
            return rew

        elif self.zero_index:
            # 2) The "old approach": landmark index=0 as special,
            #    but only agent0 gets additional_rew (the distance).

            # =========== OLD approach referencing landmark index=0 =============
            chosen_landmark_pos = state.p_pos[self.num_agents:][0]  # "leftmost" or index=0
            dist_to_chosen = jnp.linalg.norm(state.p_pos[0] - chosen_landmark_pos)
            additional_rew = -dist_to_chosen

            rew = {
                a: (
                    _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio),
                    additional_rew if i == 0 else 0.0 # Apply additional reward for agent 0
                )
                for i, a in enumerate(self.agents)
            }
            return rew

        else:
            # 3) The "new approach": find the farthest-from-agent1&2 landmark.
            #    Again only agent0 gets additional_rew.
            # =========== NEW approach: farthest from agent1 & agent2 ============
            print("---------------")
            print("farthest landmark")
            print("---------------")
            
            def min_dist_to_agents12(state: State, landmark_pos: chex.Array):
                d1 = jnp.linalg.norm(landmark_pos - state.p_pos[1])
                d2 = jnp.linalg.norm(landmark_pos - state.p_pos[2])
                return jnp.minimum(d1, d2)

            # shape => (num_landmarks,)
            dist_array = jax.vmap(lambda lm: min_dist_to_agents12(state, lm))(
                state.p_pos[self.num_agents:]
            )
            farthest_idx = jnp.argmax(dist_array)
            chosen_landmark_pos = state.p_pos[self.num_agents:][farthest_idx]

            dist_to_chosen = jnp.linalg.norm(state.p_pos[0] - chosen_landmark_pos)
            additional_rew = -dist_to_chosen

            rew = {
                a: (
                    _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio),
                    additional_rew if i == 0 else 0.0 # Apply additional reward for agent 0
                )
                for i, a in enumerate(self.agents)
            }

            return rew

            # Now that we have 'additional_rew' for agent_0, let's handle the final return:

            # ================== If self.all_agents_intrinsic is True? ==================

        #
        # index = jnp.argmin(state.p_pos[self.num_agents:, 0])
        # leftmost_landmark_pos = state.p_pos[self.num_agents:][index]

        ### === INTRINSIC REWARD FOR ALL AGENTS (USING THE 0TH LANDMARK) === ###
        # if self.all_agents_intrinsic is True:
        #     # pick the "zeroth" landmark from the array
        #     # (not necessarily the leftmost if landmarks are random, but that's your chosen logic)
        #     zero_landmark_pos = state.p_pos[self.num_agents:][0]
        #
        #     # for each agent, negative distance to that zero-th landmark
        #     @partial(jax.vmap, in_axes=(0,))
        #     def _intrinsic(aidx: int) -> float:
        #         dist = jnp.linalg.norm(state.p_pos[aidx] - zero_landmark_pos)
        #         return -dist
        #
        #     intrinsic_vals = _intrinsic(self.agent_range)
        #
        #     ### === Construct the final dictionary === ###
        #     # each agent gets (extrinsic, intrinsic)
        #     # optionally scale or manipulate the intrinsic if you wish
        #
        #     rew = {
        #         a: (
        #             _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio),
        #             intrinsic_vals[i]   # Apply additional reward for agent 0
        #         )
        #         for i, a in enumerate(self.agents)
        #     }
        #
        #     return rew
        #
        # else:
        #
        #     leftmost_landmark_pos = state.p_pos[self.num_agents:][0]
        #
        #     dist_to_leftmost_landmark = jnp.linalg.norm(state.p_pos[0] - leftmost_landmark_pos)
        #     additional_rew = -dist_to_leftmost_landmark  # Reward based on distance to the leftmost landmark
        #
        #     rew = {
        #         a: (
        #             _agent_rew(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio),
        #             additional_rew  if i == 0 else 0.0  # Apply additional reward for agent 0
        #         )
        #         for i, a in enumerate(self.agents)
        #     }
        #     return rew
