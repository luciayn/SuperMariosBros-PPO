import retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

model = PPO.load("tmp/best_model.zip")

def main():
    env = retro.make(game="SuperMarioBros-Nes")
    env = MaxAndSkipEnv(env,4)

    obs = env.reset()
    done = False

    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()
