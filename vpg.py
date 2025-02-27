import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, obs):
        return self.fc(obs)


# TODO: Inherit from classstable_baselines3.common.base_class.BaseAlgorithm
# and implement the required interface.
class VanillaPolicyGradient:
    def __init__(self, env, lr=0.01):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n  # Assuming discrete actions
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_returns_for_episode(self, rewards, gamma=0.99):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            log_probs = []
            rewards = []

            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_probs = self.policy(obs_tensor)
                action = torch.multinomial(action_probs, 1).item()
                log_probs.append(torch.log(action_probs[action]))
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                rewards.append(reward)

            returns = self.compute_returns_for_episode(rewards)
            loss = -sum(log_probs * returns)  # Policy gradient loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, observation):
        """
        Returns action based on observation
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        action_probs = self.policy(obs_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return action
