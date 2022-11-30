
from datetime import datetime
import functools
import math
import os
import pprint
import jax
import jax.numpy as jnp
#from IPython.display import HTML, clear_output
import matplotlib.pyplot as plt
import numpy as np

import brax

from brax.io import html
from brax.experimental.composer import composer
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.envs.obs_indices import OBS_INDICES
from brax.experimental.braxlines.training import ppo
from brax.experimental.braxlines.vgcrl import evaluators as vgcrl_evaluators
from brax.experimental.braxlines.vgcrl import utils as vgcrl_utils

import tensorflow_probability as tfp
from absl import app
tfp = tfp.substrates.jax
tfd = tfp.distributions



def main(argv):
    env_name = 'ant'  # @param ['ant', 'humanoid', 'halfcheetah', 'uni_ant', 'bi_ant']
    obs_indices = 'vel'  # @param ['vel']
    obs_scale = 10.0 #@param{'type': 'number'}
    obs_indices_str = obs_indices
    obs_indices = OBS_INDICES[obs_indices][env_name]

    evaluate_mi = False # @param{'type': 'boolean'}
    evaluate_lgr = False # @param{'type': 'boolean'}
    algo_name = 'diayn'  # @param ['gcrl', 'cdiayn', 'diayn', 'diayn_full', 'fixed_gcrl']
    env_reward_multiplier =   0# @param{'type': 'number'}
    obs_norm_reward_multiplier =   0# @param{'type': 'number'}
    normalize_obs_for_disc = False  # @param {'type': 'boolean'}
    seed =   0# @param {type: 'integer'}
    diayn_num_skills = 8  # @param {type: 'integer'}
    spectral_norm = True  # @param {'type': 'boolean'}
    output_path = 'ouput/run1' # @param {'type': 'string'}
    task_name = "" # @param {'type': 'string'}
    exp_name = '' # @param {'type': 'string'}
    if output_path:
        output_path = output_path.format(
            date=datetime.now().strftime('%Y%m%d'))
        task_name = task_name or f'{env_name}_{obs_indices_str}_{obs_scale}'
        exp_name = exp_name or algo_name 
        output_path = f'{output_path}/{task_name}/{exp_name}'
        print(f'output_path={output_path}')


    # @title Initialize Brax environment
    visualize = False # @param{'type': 'boolean'}

    # Create baseline environment to get observation specs
    base_env_fn = composer.create_fn(env_name=env_name)
    base_env = base_env_fn()

    # Create discriminator-parameterized environment
    disc = vgcrl_utils.create_disc_fn(algo_name=algo_name,
                    observation_size=base_env.observation_size,
                    obs_indices=obs_indices,
                    scale=obs_scale,
                    diayn_num_skills = diayn_num_skills,
                    spectral_norm=spectral_norm,
                    env=base_env,
                    normalize_obs=normalize_obs_for_disc)()
    extra_params = disc.init_model(rng=jax.random.PRNGKey(seed=seed))
    env_fn = vgcrl_utils.create_fn(env_name=env_name, wrapper_params=dict(
        disc=disc, env_reward_multiplier=env_reward_multiplier,
        obs_norm_reward_multiplier=obs_norm_reward_multiplier, 
        ))
    eval_env_fn = functools.partial(env_fn, auto_reset=False)

    # make inference functions and goals for LGR metric
    core_env = env_fn()
    params, inference_fn = ppo.make_params_and_inference_fn(
        core_env.observation_size, core_env.action_size,
        normalize_observations=True, extra_params=extra_params)
    inference_fn = jax.jit(inference_fn)
    goals = tfd.Uniform(low=-disc.obs_scale, high=disc.obs_scale).sample(
        seed=jax.random.PRNGKey(0), sample_shape=(10,))

    # Visualize
    # if visualize:
    # env = env_fn()
    # jit_env_reset = jax.jit(env.reset)
    # state = jit_env_reset(rng=jax.random.PRNGKey(seed=seed))



    #@title Training
    num_timesteps_multiplier =   6# @param {type: 'number'}
    ncols = 5 # @param{type: 'integer'}

    tab = logger_utils.Tabulator(
        output_path=f'{output_path}/training_curves.csv',
        append=False)

    # We determined some reasonable hyperparameters offline and share them here.
    n = num_timesteps_multiplier
    ppo_params = experiments.defaults.get_ppo_params(
        env_name, num_timesteps_multiplier, default='ant')
    train_fn = functools.partial(ppo.train, **ppo_params)

    times = [datetime.now()]
    plotpatterns = ['eval/episode_reward', 'losses/disc_loss', 'metrics/lgr',
                'metrics/entropy_all_', 'metrics/entropy_z_', 'metrics/mi_']

    def update_metrics_fn(num_steps, metrics, params):
        if evaluate_mi:
            metrics.update(vgcrl_evaluators.estimate_empowerment_metric(
            env_fn=env_fn, disc=disc,
            inference_fn=inference_fn, params=params,
            # custom_obs_indices = list(range(core_env.observation_size))[:30],
            # custom_obs_scale = obs_scale,
            ))
        if evaluate_lgr:
            metrics.update(vgcrl_evaluators.estimate_latent_goal_reaching_metric( 
            params=params, env_fn=env_fn, disc=disc, inference_fn=inference_fn,
            goals=goals))
    
    progress, plot, _, _ = experiments.get_progress_fn(
        plotpatterns, times, tab=tab, max_ncols=5,
        xlim=[0, train_fn.keywords['num_timesteps']],
        update_metrics_fn = update_metrics_fn,
        pre_plot_fn = plt.clf(),
        post_plot_fn = plt.savefig('out.png'))

    extra_loss_fns = dict(disc_loss=disc.disc_loss_fn) if extra_params else None
    _, params, _ = train_fn(
        environment_fn=env_fn, progress_fn=progress, extra_params=extra_params,
        extra_loss_fns=extra_loss_fns, seed=seed)

    plot(output_path=output_path)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

if __name__ == '__main__':
    app.run(main)