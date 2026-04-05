import gymnasium as gym
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
space = gym.spaces.Box(0, 255, (20, 20, 12), dtype="uint8")
print(f"Is image space: {is_image_space(space)}")
print(f"Is channels first: {is_image_space_channels_first(space)}")
