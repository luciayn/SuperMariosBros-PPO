import os
import retro
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gym import ObservationWrapper, spaces

# Define the CropImageWrapper (as in the training code)
def crop_image(image, x_start, y_start, width, height):
    cropped_image = image[y_start:y_start + height, x_start:x_start + width]
    return cropped_image

class CropImageWrapper(ObservationWrapper):
    def __init__(self, env, x_start, y_start, width, height):
        super(CropImageWrapper, self).__init__(env)
        self.x_start = x_start
        self.y_start = y_start
        self.width = width
        self.height = height

        if not isinstance(env.observation_space, spaces.Box):
            raise ValueError("The observation space must be of type spaces.Box")

        old_shape = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, old_shape[2]),
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return obs[self.y_start:self.y_start + self.height,
                   self.x_start:self.x_start + self.width, :]

# Function to wrap the environment for evaluation
def make_eval_env(env_id, x_start, y_start, width, height):
    env = retro.make(game=env_id)
    env = MaxAndSkipEnv(env, 4)  # Use same skip frames as during training
    env = CropImageWrapper(env, x_start, y_start, width, height)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

if __name__ == '__main__':
    env_id = "SuperMarioBros-Nes"
    best_model_path = "tmp/best_model.zip"  # Path to the best model

    # Ensure the model exists
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    crop_params = {'x_start': 20, 'y_start': 30, 'width': 200, 'height': 150}  # Same cropping as during training
    env = make_eval_env(env_id, **crop_params)

    # Load the best model
    model = PPO.load(best_model_path, env=env, device='cuda' if torch.cuda.is_available() else 'cpu')

    obs = env.reset()
    print("------------- Running Best Model -------------")

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)  # Use deterministic=True for evaluation
            obs, rewards, done, info = env.step(action)
            env.render()  # Render the game

            # Log current status
            print(f"Reward: {rewards}, Done: {done}, Info: {info}")

            if done:
                print("Episode finished! Resetting environment.")
                obs = env.reset()

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        env.close()
        print("Environment closed.")
