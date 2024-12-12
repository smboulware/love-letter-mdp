import gymnasium as gym
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env

# Import your custom environment
from lv_letter_env import LoveLetterEnv  # Replace with your actual environment

def train_trpo():
    # Create the environment
    env = make_vec_env(lambda: LoveLetterEnv(), n_envs=1)

    # Initialize the TRPO model
    model = TRPO(
        "MultiInputPolicy",
        env,
        verbose=1,    # Enable detailed training output
        learning_rate=0.001,  # Adjust learning rate as needed
        gamma=0.99,    # Discount factor for long-term rewards
        use_sde=False,
    )

    # Train the model
    model.learn(total_timesteps=50000)  # Adjust the number of timesteps based on computational resources

    # Save the model
    model.save("trpo_love_letter")

    print("Training complete. Model saved as 'trpo_love_letter'.")

if __name__ == "__main__":
    train_trpo()