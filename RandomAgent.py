import retro
import gym

env = retro.make(game="SuperMarioBros-Nes")
obs = env.reset() # Picture of current environment

print(obs.shape)

done = False  # done = True when Mario loses all 3 lives

while not done:
    obs, rew, done, info = env.step(env.action_space.sample())  # Returns random action that our agent is able to do, action_space = LEFT, RIGHT, JUMP,...
    env.render()

env.close()