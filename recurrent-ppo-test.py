import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

# Import your custom environment
from lv_letter_env import LoveLetterEnv  # Replace with your actual environment

def train_ppo():
    # Create the environment
    env = make_vec_env(lambda: LoveLetterEnv(), n_envs=1)

    # Initialize the PPO model
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,    # Enable detailed training output
        learning_rate=0.0003,  # Default learning rate for PPO
        gamma=0.99,    # Discount factor for long-term rewards
        n_steps=2048,  # Number of steps to collect before updating the policy
        batch_size=64,  # Batch size for training
        ent_coef=0.01,  # Encourage exploration with entropy coefficient
    )

    # Train the model
    model.learn(total_timesteps=100000)  # Adjust timesteps as needed

    # Save the model
    model.save("recurrent_ppo_love_letter_100000")

    print("Training complete. Model saved as 'recurrent_ppo_love_letter'.")

if __name__ == "__main__":
    train_ppo()