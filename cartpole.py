import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Function to build the DQN model
def build_model(states, actions):
    """
    Builds the neural network model for the DQN agent.
    
    Args:
        states (int): Number of state dimensions in the environment.
        actions (int): Number of possible actions in the environment.
    
    Returns:
        model: A compiled Keras Sequential model.
    """
    model = Sequential()
    # Input layer to flatten the state input
    model.add(Flatten(input_shape=(1, states)))
    # Two hidden layers with ReLU activation
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    # Output layer with linear activation for Q-value predictions
    model.add(Dense(actions, activation="linear"))
    return model

# Function to build the DQN agent
def build_agent(model, actions):
    """
    Builds the DQN agent with memory and policy.

    Args:
        model: The neural network model for the agent.
        actions (int): Number of possible actions in the environment.

    Returns:
        dqn: A configured DQN agent.
    """
    # Boltzmann policy to balance exploration and exploitation
    policy = BoltzmannQPolicy()
    # Sequential memory to store experiences
    memory = SequentialMemory(limit=100000, window_length=1)
    # DQN agent configuration
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,  # Initial warmup steps without training
        target_model_update=1e-2  # Frequency of target model updates
    )
    return dqn

# Function to train the DQN agent
def train_agent(agent, env, steps=50000):
    """
    Trains the DQN agent on the given environment.

    Args:
        agent: The DQN agent to be trained.
        env: The Gym environment.
        steps (int): Total training steps.

    Returns:
        None
    """
    # Compile the agent with an optimizer and metrics
    agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    # Train the agent
    agent.fit(env, nb_steps=steps, visualize=False, verbose=2)

# Function to test the DQN agent
def test_agent(agent, env, episodes=10):
    """
    Tests the trained DQN agent on the environment.

    Args:
        agent: The trained DQN agent.
        env: The Gym environment.
        episodes (int): Number of test episodes.

    Returns:
        avg_reward (float): Average reward over the test episodes.
    """
    # Evaluate the agent on the environment
    scores = agent.test(env, nb_episodes=episodes, visualize=False)
    avg_reward = np.mean(scores.history['episode_reward'])
    print(f"Average Reward over {episodes} episodes: {avg_reward}")
    return avg_reward

# Main function to orchestrate training and testing
def main(render_mode="human", train_steps=50000, test_episodes=10):
    """
    Main function to set up, train, and test the DQN agent.

    Args:
        render_mode (str): Render mode for the Gym environment.
        train_steps (int): Total training steps.
        test_episodes (int): Number of test episodes.

    Returns:
        None
    """
    try:
        # Create the CartPole-v1 environment
        env = gym.make("CartPole-v1", render_mode=render_mode)
        # Retrieve state and action space sizes
        states = env.observation_space.shape[0]
        actions = env.action_space.n

        # Build the model and the agent
        model = build_model(states, actions)
        agent = build_agent(model, actions)

        # Train the agent
        print("Training the agent...")
        train_agent(agent, env, steps=train_steps)

        # Test the agent
        print("Testing the agent...")
        test_agent(agent, env, episodes=test_episodes)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the environment
        env.close()

# Run the script
if __name__ == "__main__":
    main(render_mode="human", train_steps=50000, test_episodes=10)