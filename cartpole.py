import gym
import random


# Create env
env = gym.make("CartPole-v1", render_mode="human")

episodes = 10
for ep in range(1, episodes+1):
    state, _ = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
        state = next_state
        env.render()

    print(f"Episode {ep}, Score: {score}")


env.close()