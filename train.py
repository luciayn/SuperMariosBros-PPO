import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import os
import retro
from gym import ObservationWrapper, spaces

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

# Callback for saving the best model
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

# Function to initialize environment with cropping
def make_env(env_id, rank, seed=0, x_start=0, y_start=0, width=160, height=210):
    def _init():
        env = retro.make(game=env_id)
        env = MaxAndSkipEnv(env, 4)  # Make a decision every 4 frames
        env = CropImageWrapper(env, x_start, y_start, width, height)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Main code
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

if __name__ == '__main__':
    env_id = "SuperMarioBros-Nes"
    num_cpu = 4
    crop_params = {'x_start': 20, 'y_start': 30, 'width': 200, 'height': 150}  # Adjust cropping parameters as needed

    env = VecMonitor(
        SubprocVecEnv([make_env(env_id, i, **crop_params) for i in range(num_cpu)]),
        log_dir
    )

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003, device='cuda')

    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=10000000, callback=callback, tb_log_name="PPO-0000310M")
    model.save(env_id)
    print("------------- Done Learning -------------")

    env = retro.make(game=env_id)
