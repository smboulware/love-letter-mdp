import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

# Import your custom environment
from lv_letter_env import LoveLetterEnv  # Replace with your actual environment

def mask_fn(env: LoveLetterEnv) -> np.ndarray:
    num_card_indices = env.hidden_action_space.spaces[0].n
    num_targets = env.hidden_action_space.spaces[1].n
    num_guesses = env.hidden_action_space.spaces[2].n

    mask = np.zeros(env.action_space.nvec, dtype=int)

    # Iterate over all possible actions
    for card_index in range(num_card_indices):
        for target in range(-1, num_targets - 1):  # Adjust target indexing to match the environment
            for guess in range(num_guesses):
                # Check if the action is valid
                if env._action_valid(card_index, target, guess):
                    mask[card_index, target + 1, guess] = 1  # Add 1 to `target` to handle negative values in Discrete space

    return mask.flatten()

def train_ppo():
    
    # Create the environment
    env = LoveLetterEnv()
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # Initialize the PPO model
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,    # Enable detailed training output
        learning_rate=0.0003,  # Default learning rate for PPO
        gamma=0.99,    # Discount factor for long-term rewards
        n_steps=2048,  # Number of steps to collect before updating the policy
        batch_size=64,  # Batch size for training
        ent_coef=0.01,  # Encourage exploration with entropy coefficient
    )

    # Train the model
    model.learn(total_timesteps=50000)  # Adjust timesteps as needed

    # Save the model
    model.save("maskable_ppo_love_letter")

    print("Training complete. Model saved as 'maskable_ppo_love_letter'.")

if __name__ == "__main__":
    train_ppo()