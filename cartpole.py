import gym
import random


# Create env
env = gym.make("CartPole-v1", render_mode="human")

episods = 10
for ep in range(1, episods+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        _, reward, done =  env.step(action)
        score += reward
        env.render()

    print(f"Episode {ep}, Score: {score}")


env.close()