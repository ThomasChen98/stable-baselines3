from stable_baselines3.dqn_residual.dqn_residual import ResidualSoftDQN
from stable_baselines3.dqn_residual.policies import ResidualSoftCnnPolicy, ResidualSoftMlpPolicy, ResidualSoftMultiInputPolicy

__all__ = ["ResidualSoftCnnPolicy", "ResidualSoftMlpPolicy", "ResidualSoftMultiInputPolicy", "ResidualSoftDQN"]