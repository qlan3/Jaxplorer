import gymnasium as gym


class UniversalSeed(gym.Wrapper):
  def set_seed(self, seed: int):
    _, _ = self.env.reset(seed=seed)
    self.env.action_space.seed(seed)
    self.env.observation_space.seed(seed)