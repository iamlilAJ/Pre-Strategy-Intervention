"""
Based on PureJaxRL Implementation of PPO
"""

import os


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from jaxmarl.wrappers.baselines import LogWrapper, PrePolicyHanabiWrapper
import jaxmarl
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf

from agent.ippo_agent import PrePolicyIPPO, BaselineIPPO

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


class CustomTrainState(TrainState):
    test_returns: float = 0.0

def batchify(x: dict, agent_list):
    x = jnp.stack([x[a] for a in agent_list], axis=0)
    return x


def unbatchify(x: jnp.ndarray, agent_list):
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = PrePolicyHanabiWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        if config['RUN_BASELINE'] is True:
            network = BaselineIPPO(env.action_space(env.agents[0]).n, config=config)
            augmented_feature_size=0
        else:
            network = PrePolicyIPPO(env.action_space(env.agents[0]).n, config=config, num_agents=len(env.agents))
            augmented_feature_size=1
        rng, _rng = jax.random.split(rng)


        init_x = (
            jnp.zeros(
                (len(env.agents), 1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n +augmented_feature_size)
            ),
            jnp.zeros((len(env.agents), 1, config["NUM_ENVS"])),
            jnp.zeros((len(env.agents), 1, config["NUM_ENVS"], env.action_space(env.agents[0]).n))
        )
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                obs_batch = batchify(last_obs, env.agents)

                ac_in = (obs_batch[:, np.newaxis], last_done[:, np.newaxis], avail_actions[ :, np.newaxis])
                # print(ac_in)
                pi, value = network.apply(train_state.params, ac_in)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents)
                env_act = jax.tree_map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                # print(f"info {info}")
                info = jax.tree_map(lambda x: x.reshape((len(env.agents), config["NUM_ENVS"])), info)
                done_batch = batchify(done, env.agents).squeeze()

                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents)
            avail_actions = jnp.ones(
                (len(env.agents), config["NUM_ENVS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (last_obs_batch[:, np.newaxis], last_done[:, np.newaxis], avail_actions)
            _, last_val = network.apply(train_state.params, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params,
                                                  (traj_batch.obs, traj_batch.done, traj_batch.avail_actions))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)



                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                # print(f"batch {batch}")

                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=2), batch
                )
                #

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], x.shape[1], config["NUM_MINIBATCHES"], -1] + list(x.shape[3:])
                        ),
                        0,  # Swap the agent dimension
                        2,  # Swap with NUM_MINIBATCHES
                    ),
                    shuffled_batch,
                )
                # print(f"minibatches {minibatches}")
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            train_state = train_state.replace(
                test_returns=train_state.test_returns + metric["returned_episode_extrinsic_returns"][-1, :].mean() /10.
            )
            def callback(metric, train_state):
                wandb.log(
                    {
                        "mixed_returns": metric["returned_episode_mixed_returns"][-1, :].mean(),
                        "extrinsic_returns": metric["returned_episode_extrinsic_returns"][-1, :].mean(),
                        "intrinsic_returns": metric["returned_episode_intrinsic_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        "accumulated_extrinsic_return": train_state.test_returns,

                    }
                )
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric, train_state)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv,
                        jnp.zeros((len(env.agents), (config["NUM_ENVS"])), dtype=bool), _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

def single_run(config):
    config = {**config, **config["alg"]}  # merge the baselines config with the main config

    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN"],
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config), device=jax.devices()[0])
    out = train_jit(rng)

    # Save params
    if config['SAVE_PATH'] is not None:

        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        params = out['runner_state'][0][0].params
        save_dir = os.path.join(config['SAVE_PATH'], run.project, run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/model.safetensors')

        # upload this to wandb as an artifact
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()


def tune(default_config):
    default_config = {**default_config, **default_config["alg"]}  # merge the baselines config with the main config

    """Hyperparameter sweep with wandb."""
    def flatten_dict(d, parent_key='', sep='.'):
        """Flatten a nested dictionary for WandB compatibility."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def unflatten_dict(d, sep='.'):
        """Reconstruct a nested dictionary from a flattened one."""
        result = {}
        for k, v in d.items():
            keys = k.split(sep)
            target = result
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = v
        return result

    def wrapped_make_train():
        import copy
        wandb.init(project=default_config["PROJECT"],
                   tags=["IPPO", "hanabi"],)

        # update the default params
        config = copy.deepcopy(default_config)

        flattened_config = flatten_dict(config)

        for k, v in dict(wandb.config).items():
            flattened_config[k] = v
        config = unflatten_dict(flattened_config)

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])

        train_vjit = jax.jit(make_train(config),  device=jax.devices()[0])
        outs = train_vjit(rng)

    if default_config['RUN_BASELINE'] is True:
        alg_name = "baseline_no_sharing_ippo"
        config_name = ""
    else:
        alg_name = "pre_policy_ippo"
        config_name = default_config["CONFIG_NAME"]


    sweep_config = {
        "name": f"{alg_name}_config_{config_name}_hanabi",
        "method": "grid",
        "metric": {
            "name": "accumulated_extrinsic_return",
            "goal": "maximize",
        },

        'parameters': {

            "LR": {
                "values": [0.0005, 0.0006, 0.0007]

            }

        }
    }

    wandb.login()

    sweep_id = default_config.get("SWEEP_ID", None)

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"])
    else:
        print(f"Using existing sweep ID: {sweep_id}")

    wandb.agent(sweep_id, wrapped_make_train, count=300, entity=default_config["ENTITY"], project=default_config["PROJECT"])


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()