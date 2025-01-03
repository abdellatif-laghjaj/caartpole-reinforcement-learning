import gym
import random

from rl import agents
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# Create env
env = gym.make("CartPole-v1", render_mode="human")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

agent = DQNAgent(
        model=model,
        memory=SequentialMemory(limit=50000, window_length=1),
        policy=BoltzmannQPolicy(),
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2
    )

agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
agent.fit(env, nb_steps=50000, visualize=False, verbose=2)

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