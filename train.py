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

class SaveOnBestTrainingRewardCallback(BaseCallback): # [OPTIONAL]
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
    
def make_env(env_id, rank, seed=0):
    def _init ():
        env = retro.make(game=env_id)
        env = MaxAndSkipEnv(env, 4)  # Make a decision every 4 frames, help the training and learning
        env.seed(seed + rank)  # Random environment
        return env
    set_random_seed(seed)
    return _init

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

if __name__ == '__main__':
    env_id = "SuperMarioBros-Nes"
    num_cpu = 4

    # Mushing all the different environments into a single list and passing it to SubprocVecEnv, which we are wrapping into our VecMonitor so we can keep track of it while it's training
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]),"tmp/monitor")

    # Color images RGB, we use CNNs that are able to find features within images
    # verbose: info messages
    # learning_rate: adjust by looking the tensorboard
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", learning_rate=0.00003)
    # model = PPO.load("path_to_model", env=env)

    print("------------- Start Learning -------------")
    # Sve the model everytime, when to stop or when our agent is good enough
    # Check every 1000 steps, and save to log_dir
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=1000000, callback=callback, tb_log_name="PPO-00003")  # Store each experiment with specific name
    model.save(env_id)  # Might not be the best model, could downgrade over time, but we will store it just in case
    print("------------- Done Learning -------------")
    env = retro.make(game=env_id)
