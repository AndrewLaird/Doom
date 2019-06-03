import gym
import vizdoomgym

if __name__ == "__main__":
    env = gym.make("VizdoomBasic-v0")

    observation = env.reset()
    for i in range(20000):
        env.render()
        action =  env.action_space.sample()
        stuff = env.step(action)
