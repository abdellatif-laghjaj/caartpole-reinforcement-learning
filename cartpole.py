import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    """Builds the DQN model."""
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model

def build_agent(model, actions):
    """Creates the DQN agent."""
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2,
    )
    return dqn

def train_agent(agent, env, steps=50000):
    """Trains the DQN agent."""
    agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    agent.fit(env, nb_steps=steps, visualize=False, verbose=2)

def test_agent(agent, env, episodes=10):
    """Tests the trained DQN agent."""
    scores = agent.test(env, nb_episodes=episodes, visualize=False)
    avg_reward = np.mean(scores.history['episode_reward'])
    print(f"Average Reward over {episodes} episodes: {avg_reward}")
    return avg_reward

def main(render_mode="human", train_steps=50000, test_episodes=10):
    # Create the environment
    env = gym.make("CartPole-v1", render_mode=render_mode)
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    # Build the model and agent
    model = build_model(states, actions)
    agent = build_agent(model, actions)

    # Train and test the agent
    print("Training the agent...")
    train_agent(agent, env, steps=train_steps)

    print("Testing the agent...")
    test_agent(agent, env, episodes=test_episodes)

    env.close()

if __name__ == "__main__":
    main(render_mode="human", train_steps=50000, test_episodes=10)