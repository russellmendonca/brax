import warnings
warnings.filterwarnings('ignore', message='.*tree_map', )
warnings.filterwarnings('ignore', message='.*Unable to initialize backend', )
warnings.filterwarnings('ignore', message='.*Explicitly requested dtype.*int64.* requested in astype is not available', )

# For use on cluster
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # specify which GPU(s) to be used

from datetime import datetime
import functools
import math
import os
import pprint
import jax
import jax.numpy as jnp
import pickle
#from IPython.display import HTML, clear_output
import matplotlib.pyplot as plt
import numpy as np

import brax

from brax.io import html, image
from brax.experimental.composer import composer
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.envs.obs_indices import OBS_INDICES
from brax.experimental.braxlines.training import ppo
from brax.experimental.braxlines.vgcrl import evaluators as vgcrl_evaluators
from brax.experimental.braxlines.vgcrl import utils as vgcrl_utils
from brax.experimental.braxlines.vgcrl.utils import FixedSkillWrapper

import tensorflow_probability as tfp
from absl import app
tfp = tfp.substrates.jax
tfd = tfp.distributions



def main(argv):
    env_name = 'ant'  # @param ['ant', 'humanoid', 'halfcheetah', 'uni_ant', 'bi_ant']
    disable_action_entropy = False # False
    obs_indices = 'vel'  # @param ['vel']
    obs_scale = 10.0 #@param{'type': 'number'}
    obs_indices_str = obs_indices
    obs_indices = OBS_INDICES[obs_indices][env_name]

    # mutual information
    # evaluate_mi = False # @param{'type': 'boolean'}
    # evaluate_lgr = False # @param{'type': 'boolean'}
    algo_name = 'diayn'  # @param ['gcrl', 'cdiayn', 'diayn', 'diayn_full', 'fixed_gcrl']
    env_reward_multiplier =   0# @param{'type': 'number'}
    obs_norm_reward_multiplier =   0# @param{'type': 'number'}
    normalize_obs_for_disc = False  # @param {'type': 'boolean'}
    seed =   0# @param {type: 'integer'}
    diayn_num_skills = 8  # @param {type: 'integer'}
    # NOTE: what is spectral norm?
    spectral_norm = True  # @param {'type': 'boolean'}
    output_path = 'output/run1' # @param {'type': 'string'}
    task_name = "" # @param {'type': 'string'}
    exp_name = algo_name # @param {'type': 'string'}
    if disable_action_entropy:
        exp_name += '_no_a_ent'
    num_timesteps_multiplier =   6# @param {type: 'number'}
    if output_path:
        output_path = output_path.format(
            date=datetime.now().strftime('%Y%m%d'))
        task_name = task_name or f'{env_name}_{obs_indices_str}_{obs_scale}'
        exp_name = exp_name or algo_name 
        output_path = f'{output_path}/{task_name}/{exp_name}'
        print(f'output_path={output_path}')


    # Create baseline environment to get observation specs
    base_env_fn = composer.create_fn(env_name=env_name)
    # Create discriminator-parameterized environment
    disc = vgcrl_utils.create_disc_fn(algo_name=algo_name,
                    observation_size=base_env_fn().observation_size,
                    obs_indices=obs_indices,
                    scale=obs_scale,
                    diayn_num_skills = diayn_num_skills,
                    spectral_norm=spectral_norm,
                    env=base_env_fn(),
                    normalize_obs=normalize_obs_for_disc)()
    extra_params = disc.init_model(rng=jax.random.PRNGKey(seed=seed))
    disc_env_fn = vgcrl_utils.create_fn(env_name=env_name, wrapper_params=dict(
        disc=disc, env_reward_multiplier=env_reward_multiplier,
        obs_norm_reward_multiplier=obs_norm_reward_multiplier, 
        ))

    #@title Training

    # We determined some reasonable hyperparameters offline and share them here.
    ppo_params = experiments.defaults.get_ppo_params(
        env_name, num_timesteps_multiplier, default='ant')
    if disable_action_entropy:
        ppo_params['entropy_cost'] = 0
    print('PPO parameters:', ppo_params)
    train_fn = functools.partial(ppo.train, **ppo_params)

    tab = logger_utils.Tabulator(
        output_path=f'{output_path}/training_curves.csv',
        append=False)
    times = [datetime.now()]
    plotpatterns = ['eval/episode_reward', 'losses/disc_loss', 'metrics/lgr',
                'metrics/entropy_all_', 'metrics/entropy_z_', 'metrics/mi_']
    progress, plot, _, _ = experiments.get_progress_fn(
        plotpatterns, times, tab=tab, max_ncols=5,
        xlim=[0, train_fn.keywords['num_timesteps']],
        pre_plot_fn = plt.clf,)

    _, params, _ = train_fn(
        environment_fn=disc_env_fn, progress_fn=progress, extra_params=extra_params,
        extra_loss_fns=dict(disc_loss=disc.disc_loss_fn), seed=seed)

    plot(output_path=output_path)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    with open(f'{output_path}/params.pkl', 'wb') as f:
        pickle.dump(params, f)


    eval_env_fn = functools.partial(disc_env_fn, auto_reset=False)
    eval_env = eval_env_fn()
    inference_fn = ppo.make_inference_fn(
        eval_env.observation_size, eval_env.action_size,
        normalize_observations=True)

    num_z = 5  # @param {type: 'integer'}
    num_samples_per_z = 5  # @param {type: 'integer'}
    time_subsampling = 10  # @param {type: 'integer'}
    time_last_n = 500 # @param {type: 'integer'}
    eval_seed = 0  # @param {type: 'integer'}

    vgcrl_evaluators.visualize_skills(
        env_fn=eval_env_fn,
        disc=disc,
        inference_fn=inference_fn,
        params=params,
        output_path=output_path,
        verbose=True,
        num_z=num_z,
        num_samples_per_z=num_samples_per_z,
        time_subsampling=time_subsampling,
        time_last_n=time_last_n,
        save_video=True,
        seed=eval_seed)
    plt.clf()

    eval_seed = 0  # @param {‘type’: ‘integer’}
    frames = 5

    for z_value in range(diayn_num_skills):
        z = jax.nn.one_hot(jnp.array(int(z_value)), disc.z_size)
        eval_env, states = evaluators.visualize_env(
            env_fn=eval_env_fn,
            inference_fn=inference_fn,
            params=params,
            batch_size=0,
            seed = eval_seed,
            reset_args = (z,),
            step_args = (params['normalizer'], params['extra']),
            output_path=output_path,
            output_name=f'video_z_{z_value}',
        )
        f, axs = plt.subplots(1, frames, figsize=(10*frames, 8))
        for ax, si in zip(axs, np.round(np.linspace(0, len(states)-1, frames)).astype(int)):
            ax.imshow(image.render_array(eval_env.sys, states[si].qp, width=640, height=480))
            ax.set_axis_off()
        plt.savefig(f'{output_path}/skill_{z_value}.png', bbox_inches='tight', pad_inches = 0)

    # Find best skill
    eval_seed = 0  # @param {‘type’: ‘integer’}

    skill_return = []
    for z_value in range(diayn_num_skills):
        z = jax.nn.one_hot(jnp.array(int(z_value)), disc.z_size)
        eval_env, states = evaluators.visualize_env(
            env_fn=eval_env_fn,
            inference_fn=inference_fn,
            params=params,
            batch_size=0,
            seed = eval_seed,
            reset_args = (z,),
            output_path=output_path,
            output_name=f'video_z_{z_value}',
            step_fn_name="step2",
        )
        skill_return.append(sum([s.reward.item() for s in states]))
    best_skill = np.argmax(skill_return)
    print(f'Found best skill {best_skill}')

    # Finetune
    tab = logger_utils.Tabulator(
        output_path=f'{output_path}/training_curves_ft.csv',
        append=False)
    progress, plot, _, _ = experiments.get_progress_fn(
        plotpatterns, times, tab=tab, max_ncols=5,
        xlim=[0, train_fn.keywords['num_timesteps']],
        pre_plot_fn = plt.clf)
    fixed_skill_env_fn = lambda **kwargs: FixedSkillWrapper(
        composer.create(env_name=env_name, **kwargs),
        z=jax.nn.one_hot(jnp.array(best_skill), disc.z_size),
        disc=disc)
    _, ft_params, _ = train_fn(
        environment_fn=fixed_skill_env_fn, progress_fn=progress, seed=seed, policy_params=params['policy'])
    plot(output_path=output_path, output_name='training_curves_ft')

    with open(f'{output_path}/ft_params.pkl', 'wb') as f:
        pickle.dump(ft_params, f)


if __name__ == '__main__':
    app.run(main)