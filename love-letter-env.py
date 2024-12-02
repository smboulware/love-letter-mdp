import gym
from gym import spaces
import numpy as np

class LoveLetter(gym.Env):
    """Custom POMDP Environment"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(LoveLetter, self).__init__()
        num_cards = 8
        seed = 184
        # Define action and observation spaces
        
        # Actions: card, target, and guessed card (any card except guard)
        self.action_space = spaces.Dict({"card": spaces.Discrete(num_cards), "target": spaces.Discrete(5), "guessed card": spaces.Discrete(num_cards - 1)}, seed=seed)
        
        # Observation: Each player's card (0 for eliminated), remaining cards
        self.observation_space = spaces.Dict({"p1 card" : spaces.Discrete(num_cards + 1), "p2 card" : spaces.Discrete(num_cards + 1), 
            "p3 card" : spaces.Discrete(num_cards + 1), "p4 card" : spaces.Discrete(num_cards + 1), 
            "remaining" : spaces.MultiDiscrete(np.full(num_cards, 2))}, seed=seed)
        
        # Hidden state (true position of the agent, not directly observable)
        self.state = None
        
        # Target state
        self.target_state = 100
        
        # Step counter
        self.step_count = 0
        self.max_steps = 50  # TODO: max number of turns in the game
    
    def step(self, action):
        """
        Executes a step in the environment.
        """
        self.step_count += 1
        
        # Update the true state based on the action
        # TODO save actions of other players
        if action == 0:  # Move left
            self.state -= 1
        elif action == 2:  # Move right
            self.state += 1
        
        # Ensure the state stays within bounds
        self.state = np.clip(self.state, 0, 100)
        
        # Generate a partial observation (noisy position)
        noise = np.random.normal(0, 5)  # Add Gaussian noise with std=5
        observation = np.array([self.state + noise], dtype=np.float32)
        
        # Calculate the reward
        reward = -abs(self.target_state - self.state)  # Negative distance to the target
        
        # Check if the episode is done
        done = self.state == self.target_state or self.step_count >= self.max_steps
        
        # Additional info for debugging
        info = {
            "true_state": self.state  # Include the true state for debugging or evaluation
        }
        
        return observation, reward, done, info
    
    def reset(self):
        """
        Resets the environment to an initial state.
        """
        self.state = np.random.randint(0, 100)  # Random starting position
        self.step_count = 0
        
        # Generate the initial noisy observation
        noise = np.random.normal(0, 5)
        observation = np.array([self.state + noise], dtype=np.float32)
        
        return observation
    
    def render(self, mode='human'):
        """
        Renders the environment.
        """
        print(f"True state: {self.state}, Target: {self.target_state}")
    
    def close(self):
        """
        Cleanup when the environment is closed.
        """
        pass
