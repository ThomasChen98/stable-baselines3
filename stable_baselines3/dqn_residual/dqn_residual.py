import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import gymnasium as gym
import highway_env
highway_env.register_highway_envs()
from highway_env import utils
from highway_env.vehicle.controller import ControlledVehicle
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn_residual.policies import ResidualSoftDQNPolicy, ResidualSoftMlpPolicy, ResidualSoftCnnPolicy, ResidualSoftMultiInputPolicy, ResidualSoftQNetwork
from stable_baselines3.dqn_me.dqn_me import DQN_ME

SelfResidualSoftDQN = TypeVar("SelfResidualSoftDQN", bound="ResidualSoftDQN")


class ResidualSoftDQN(DQN_ME):
    """
    Residual Soft Deep Q-Network

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236, https://arxiv.org/abs/1702.08165     
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ResidualSoftMlpPolicy,
        "CnnPolicy": ResidualSoftCnnPolicy,
        "MultiInputPolicy": ResidualSoftMultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: ResidualSoftQNetwork
    q_net_target: ResidualSoftQNetwork
    policy: ResidualSoftDQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[ResidualSoftDQNPolicy]],
        env: Union[GymEnv, str],
        prior_model_path: str,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        env_update_freq: int = 1000,
        theta: List[float] = [0., 0., 0.],
        eta: float = 0.5,
        expert_expected_feature_count: List[float] = [0., 0., 0.],
        epsilon: float = 0.05,
        sample_length: int = 1000,
        ignore_prior: bool = False,
    ):
        self.prior_model_path = prior_model_path
        self.env_update_freq = env_update_freq
        self.theta = theta
        self.eta = eta
        self.expert_expected_feature_count = expert_expected_feature_count
        self.epsilon = epsilon
        self.sample_length = sample_length
        self.ignore_prior = ignore_prior
        policy_kwargs['ignore_prior'] = ignore_prior
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
    
    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy.prior_model = DQN_ME.load(self.prior_model_path)
        self.policy.prior_model.policy.set_training_mode(False) # freeze prior model parameters
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Compute the prior log probability
                prior_logprob = self.policy.prior_model.policy.predict_logprob(replay_data.next_observations)
                if self.ignore_prior: # ignore prior_logprob
                    next_q_values = th.logsumexp(next_q_values, 1)
                else:
                    next_q_values = th.logsumexp(next_q_values + prior_logprob, 1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def learn(
        self: SelfResidualSoftDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SelfResidualSoftDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfResidualSoftDQN:

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # update env
            if self.num_timesteps > self.learning_starts+self.env_update_freq \
                    and self.num_timesteps % self.env_update_freq == 0:
                print(f'\tInner loop timestep: {self.num_timesteps}')
                # get theta gradient
                grad_theta = self.grad_theta(length=self.sample_length)

                grad_theta_norm = np.linalg.norm(grad_theta)
                print(f"\t{'Theta Gradient:':<30}{grad_theta}\tNorm: {grad_theta_norm:.5f}")

                if grad_theta_norm < self.epsilon:
                    print(f'\tTheta converged')
                    break # gradient theta converges

                self.theta += self.eta * grad_theta
                print(f"\t{'Updated Theta:':<30}{self.theta}")
                print('\t'+'·'*85)

                new_env = gym.make('highway-addLinearReward-v0', theta=self.theta)
                self.set_env(new_env)
                self._last_obs = self.env.reset()

            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def grad_theta(self, length=200):
        trajectory = []

        print(f'\tCollecting prior trajectory with Theta = {self.theta}')
        env = gym.make('highway-addLinearReward-v0', theta=self.theta)
        obs, _ = env.reset()
        count = 0
        for _ in range(length):
            action, _ = self.predict(obs, deterministic=True)

            obs, reward, done, truncated, infos = env.step(int(action))

            # state calculation
            neighbours = env.road.network.all_side_lanes(env.vehicle.lane_index)
            lane = env.vehicle.target_lane_index[2] if isinstance(env.vehicle, ControlledVehicle) \
                else env.vehicle.lane_index[2]
            lane = lane / max(len(neighbours) - 1, 1)

            forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)
            lateral_speed = env.vehicle.speed * np.sin(env.vehicle.heading)
            scaled_speed = utils.lmap(forward_speed, [20, 30], [0, 1])

            # feature prime
            feature_prime = np.zeros((3,), dtype=float)  # feature: collision, right_lane, high_speed

            feature_prime[0] = env.vehicle.crashed
            feature_prime[1] = lane
            feature_prime[2] = scaled_speed

            trajectory.append(np.array(feature_prime))

            if done:
                obs, _ = env.reset()

            count += 1
        
        prior_expected_feature_count = np.mean(trajectory, axis = 0)
        print(f"\t{'Prior feature count:':<30}{prior_expected_feature_count}")

        return self.expert_expected_feature_count - prior_expected_feature_count