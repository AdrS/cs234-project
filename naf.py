import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from typing import Optional, Union, Tuple, Any, Dict

import gymnasium as gym

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.policies import BasePolicy
from gym.spaces import Box

BoxTypes = (gym.spaces.Box,)


def flatten_dict_observation(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    arrays = []
    for k in sorted(obs_dict.keys()):
        arr = obs_dict[k]
        arrays.append(arr.reshape(-1))
    return np.concatenate(arrays, axis=0)


def flatten_batch_of_dicts(obs_dict_batch: Dict[str, np.ndarray]) -> np.ndarray:
    keys = sorted(obs_dict_batch.keys())
    arrays = []
    for k in keys:
        arr = obs_dict_batch[k]
        arr = arr.reshape(arr.shape[0], -1)
        arrays.append(arr)
    cat = np.concatenate(arrays, axis=1)
    return cat


class DummyNAFPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space)
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        pass

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        return None, None

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        batch_size = observation.shape[0]
        return torch.zeros(
            (batch_size, self.action_space.shape[0]), device=observation.device
        )


class NAFNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.value = nn.Linear(hidden_size, 1)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.L = nn.Linear(hidden_size, action_dim * (action_dim + 1) // 2)

    def forward(self, obs, action):
        batch_size = obs.shape[0]
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))

        V = self.value(x)
        mu = self.mu(x)
        L_params = self.L(x)

        L = torch.zeros(batch_size, self.action_dim, self.action_dim, device=obs.device)
        idx = 0
        for i in range(self.action_dim):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = torch.exp(L_params[:, idx])
                else:
                    L[:, i, j] = L_params[:, idx]
                idx += 1

        P = torch.bmm(L, L.transpose(1, 2))

        diff = (action - mu).unsqueeze(2)
        A = -0.5 * torch.bmm(diff.transpose(1, 2), torch.bmm(P, diff))
        A = A.squeeze(-1).squeeze(-1)

        Q = V.squeeze(-1) + A
        Q = Q.unsqueeze(-1)
        return Q, V, mu, P


class NAF(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, nn.Module],
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 1_000,
        batch_size: int = 64,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        replay_buffer_class=None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        device: Union[str, torch.device] = "auto",
        support_multi_env: bool = True,
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        _init_setup_model: bool = True,
        **kwargs,
    ):
        if isinstance(policy, str):
            policy = DummyNAFPolicy

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            seed=seed,
            tensorboard_log=tensorboard_log,
            sde_support=False,
            **kwargs,
        )

        self.q_net = None
        self.q_net_target = None
        self.optimizer = None
        self.wandb_run = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        e = self.env
        if hasattr(e, "envs") and len(e.envs) > 0:
            e = e.envs[0]

        obs_space = e.observation_space
        act_space = e.action_space

        if isinstance(obs_space, gym.spaces.Dict):
            total_dim = 0
            for space in obs_space.spaces.values():
                total_dim += int(np.prod(space.shape))
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
            )
        else:
            self.observation_space = obs_space

        if not isinstance(act_space, BoxTypes):
            raise ValueError(
                f"NAF only supports continuous actions, got {type(act_space).__name__}"
            )

        self.action_space = act_space

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))

        self.q_net = NAFNetwork(obs_dim, act_dim)
        self.q_net_target = NAFNetwork(obs_dim, act_dim)
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.q_net.to(self.device)
        self.q_net_target.to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        if wandb.run is not None:
            self.wandb_run = wandb.run

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self._update_learning_rate(self.optimizer)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            obs_data = replay_data.observations
            next_obs_data = replay_data.next_observations

            if isinstance(obs_data, dict):
                obs_np = flatten_batch_of_dicts(obs_data)
            else:
                obs_np = obs_data
            if isinstance(next_obs_data, dict):
                next_obs_np = flatten_batch_of_dicts(next_obs_data)
            else:
                next_obs_np = next_obs_data

            obs_torch = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
            next_obs_torch = torch.tensor(
                next_obs_np, dtype=torch.float32, device=self.device
            )

            actions = replay_data.actions.float().to(self.device)
            actions = actions.view(batch_size, -1)

            rewards = replay_data.rewards.view(-1, 1).float().to(self.device)
            dones = replay_data.dones.view(-1, 1).float().to(self.device)

            obs_torch = obs_torch.view(batch_size, -1)
            next_obs_torch = next_obs_torch.view(batch_size, -1)

            Q_current, _, _, _ = self.q_net(obs_torch, actions)

            with torch.no_grad():
                dummy_next_act = torch.zeros(
                    (batch_size, self.action_space.shape[0]),
                    dtype=actions.dtype,
                    device=actions.device,
                )

                _, _, mu_next, _ = self.q_net_target(next_obs_torch, dummy_next_act)
                Q_next, _, _, _ = self.q_net_target(next_obs_torch, mu_next)

                Q_target = rewards + (1 - dones) * self.gamma * Q_next

            loss = nn.MSELoss()(Q_current, Q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            polyak_update(
                self.q_net.parameters(), self.q_net_target.parameters(), self.tau
            )

            if self.verbose > 0 and self.wandb_run is not None:
                wandb.log({"loss": loss.item()})

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if isinstance(observation, dict):
            obs_vec = flatten_dict_observation(observation)
        else:
            obs_vec = observation

        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            dummy_action = torch.zeros(
                (obs_tensor.shape[0], self.action_space.shape[0]),
                device=self.device,
                dtype=torch.float32,
            )
            Q, _, mu, _ = self.q_net(obs_tensor, dummy_action)

        return mu.cpu().numpy(), None
