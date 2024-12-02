import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# ------------------------------
# LoveLetterEnv Class Definition
# ------------------------------
class LoveLetterEnv(gym.Env):
    # Card target requirements: True if the card requires selecting a target player, False otherwise
    CARD_TARGET_REQUIREMENTS = {
        1: True,  # Guard
        2: True,  # Priest
        3: True,  # Baron
        4: False, # Handmaid
        5: True,  # Prince
        6: True,  # King
        7: False, # Countess
        8: False, # Princess
    }

    def __init__(self):
        super(LoveLetterEnv, self).__init__()

        self.num_players = 2
        self.num_cards = 16
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),  # Card index to play
            spaces.Discrete(self.num_players),  # Target player
        ))
        self.observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([8, 8]),  # Two cards in hand
            "public_state": spaces.Box(low=0, high=1, shape=(self.num_players,), dtype=np.int32),
        })

        # Initialize game state
        self.reset()

    def reset(self):
        # Initialize the deck
        self.deck = [1] * 5 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 2 + [6] * 1 + [7] * 1 + [8] * 1
        random.shuffle(self.deck)

        # Each player draws two cards
        self.hands = [[], []]
        for _ in range(2):
            for player in range(self.num_players):
                self.hands[player].append(self.deck.pop())

        # Initialize discard piles and protection statuses
        self.discard_piles = [[] for _ in range(self.num_players)]
        self.protected = [False] * self.num_players
        self.current_player = 0
        self.winner = None  # Track the winner

        return self._get_observation(), {}

    def step(self, action):
        # Unpack the action
        card_index, target = action

        # Reset protection statuses at the start of each turn
        self.protected = [False] * self.num_players

        # Play the chosen card and get feedback
        played_card = self.hands[self.current_player].pop(card_index)
        self.discard_piles[self.current_player].append(played_card)
        reward, feedback, terminated = self._apply_action(played_card, target)

        # If the game is terminated immediately, return the result
        if terminated:
            self.winner = 0 if len(self.hands[0]) > 0 else 1  # Determine the winner
            return self._get_observation(), reward, terminated, False, {"feedback": feedback, "winner": self.winner}

        # Draw a new card if the deck is not empty
        if self.deck:
            self.hands[self.current_player].append(self.deck.pop())

        # Advance to the next player
        self.current_player = (self.current_player + 1) % self.num_players

        # Check for truncation (deck runs out)
        truncated = len(self.deck) == 0
        if truncated:
            self.winner = self._determine_truncated_winner()
            return self._get_observation(), reward, False, truncated, {"feedback": feedback, "winner": self.winner}

        # Return updated observation, reward, and status flags
        return self._get_observation(), reward, False, False, {"feedback": feedback}

    def _determine_truncated_winner(self):
        """Determine the winner when the game ends due to truncation."""
        human_card = self.hands[0][0] if self.hands[0] else -1
        ai_card = self.hands[1][0] if self.hands[1] else -1
        return 0 if human_card > ai_card else 1  # Compare card values

    def _is_game_terminated(self):
        """Check if the game is over."""
        active_players = [hand for hand in self.hands if len(hand) > 0]
        return len(active_players) <= 1

    def _apply_action(self, card, target):
        """Apply the effect of the played card."""
        feedback = ""
        terminated = False

        if card == 1:  # Guard
            guessed_card = self._get_guess()
            if not self.protected[target] and guessed_card in self.hands[target]:
                self.hands[target].remove(guessed_card)  # Target is eliminated
                feedback = f"Player {target} is eliminated because their card ({self._get_card_name(guessed_card)}) was guessed."
                terminated = self._is_game_terminated()  # Check termination immediately
            else:
                feedback = f"The guess ({self._get_card_name(guessed_card)}) was incorrect."

        elif card == 2:  # Priest
            if not self.protected[target]:
                revealed_card = self.hands[target][0] if self.hands[target] else None
                feedback = f"Player {target}'s hand is revealed: {self._get_card_name(revealed_card)}."
            else:
                feedback = f"Player {target} is protected by Handmaid."

        elif card == 3:  # Baron
            if not self.protected[target]:
                if self.hands[self.current_player][0] > self.hands[target][0]:
                    eliminated_card = self.hands[target].pop(0)  # Target is eliminated
                    feedback = f"Player {target} is eliminated by Baron. Their card ({self._get_card_name(eliminated_card)}) was weaker."
                else:
                    eliminated_card = self.hands[self.current_player].pop(0)  # Current player is eliminated
                    feedback = f"You are eliminated by Baron. Your card ({self._get_card_name(eliminated_card)}) was weaker."
                terminated = self._is_game_terminated()  # Check termination immediately
            else:
                feedback = f"Player {target} is protected by Handmaid."

        elif card == 4:  # Handmaid
            self.protected[self.current_player] = True
            feedback = "You are protected from all effects until your next turn."

        elif card == 5:  # Prince
            if self.protected[target]:
                feedback = f"Player {target} is protected by Handmaid."
            else:
                discarded_card = self.hands[target].pop(0) if self.hands[target] else None
                if discarded_card == 8:  # Princess discarded
                    feedback = f"Player {target} discarded the Princess and is eliminated!"
                    self.hands[target] = []  # Eliminate the player
                    terminated = self._is_game_terminated()  # Check termination immediately
                else:
                    new_card = self.deck.pop() if self.deck else None
                    self.hands[target].append(new_card)
                    feedback = f"Player {target} discarded {self._get_card_name(discarded_card)} and drew {self._get_card_name(new_card)}."

        elif card == 6:  # King
            if not self.protected[target]:
                self.hands[self.current_player], self.hands[target] = self.hands[target], self.hands[self.current_player]
                feedback = f"You swapped hands with Player {target}."
            else:
                feedback = f"Player {target} is protected by Handmaid."

        elif card == 7:  # Countess
            feedback = "Countess is discarded. No additional effect."

        elif card == 8:  # Princess
            self.hands[self.current_player] = []  # Current player eliminated
            feedback = "You are eliminated for discarding the Princess."
            terminated = self._is_game_terminated()  # Check termination immediately

        return 0, feedback, terminated

    def _get_guess(self):
        """Determine the guessed card based on the current player."""
        if self.current_player == 0:  # Human player's turn
            while True:
                try:
                    guess = int(input("Guess the target player's card (2-8): "))
                    if 2 <= guess <= 8:
                        return guess
                    else:
                        print("Invalid guess! Please enter a number between 2 and 8.")
                except ValueError:
                    print("Invalid input! Please enter a valid number.")
        else:  # AI agent's turn
            return random.randint(2, 8)  # AI makes a random guess

    def _get_observation(self):
        """Get the current player's observation."""
        return {
            "hand": np.array(self.hands[self.current_player], dtype=np.int32),
            "public_state": np.array(self.protected, dtype=np.int32),
        }

    def _get_card_name(self, card_id):
        """Get the card name from the card ID."""
        card_details = {
            1: "Guard", 2: "Priest", 3: "Baron", 4: "Handmaid",
            5: "Prince", 6: "King", 7: "Countess", 8: "Princess"
        }
        return card_details.get(card_id, "Unknown")


# ------------------------------
# Interactive Gameplay Loop
# ------------------------------
def display_card_details():
    """Display card names and effects."""
    print("\n--- Love Letter Card Details ---")
    card_details = {
        1: "Guard: Guess another player's card (except Guard). If correct, they are eliminated.",
        2: "Priest: Look at another player's hand.",
        3: "Baron: Compare hands with another player. Lower card is eliminated.",
        4: "Handmaid: Protect yourself from effects until your next turn.",
        5: "Prince: Choose a player (or yourself) to discard their hand and draw a new card.",
        6: "King: Trade hands with another player.",
        7: "Countess: Must discard if you hold a King or Prince.",
        8: "Princess: If you discard this card, you are eliminated.",
    }
    for card_id, effect in card_details.items():
        print(f"{card_id}: {effect}")
    print("--------------------------------\n")

def human_vs_random_ai():
    """Main gameplay loop for human vs. AI."""
    # Initialize the environment
    env = LoveLetterEnv()
    obs, _ = env.reset()

    # Display card details at the start
    display_card_details()
    print("Welcome to Love Letter! Your goal is to either eliminate AI-1 or have the highest card at the end.\n")

    while True:  # Run until the game ends
        print("\n" + "-" * 50)  # Separation between turns

        if env.current_player == 0:  # Human player's turn
            print("\nYour Turn!")
            print(f"Your hand: {[f'{i}: {env._get_card_name(c)}' for i, c in enumerate(obs['hand'])]}")
            print(f"Public state (protection status): {['Protected' if p else 'Unprotected' for p in obs['public_state']]}")
            print("Players: 0: You, 1: AI-1")

            # Select card
            while True:
                try:
                    card_index = int(input(f"Choose a card to play ({', '.join([f'{i}: {env._get_card_name(c)}' for i, c in enumerate(obs['hand'])])}): "))
                    if 0 <= card_index < len(obs['hand']):
                        break
                except ValueError:
                    print("Invalid input! Try again.")

            # Determine if target selection is needed
            selected_card = obs['hand'][card_index]
            if LoveLetterEnv.CARD_TARGET_REQUIREMENTS[selected_card]:
                while True:
                    try:
                        target = int(input(f"Choose a target player (0: You, 1: AI-1): "))
                        if 0 <= target < env.num_players:
                            break
                    except ValueError:
                        print("Invalid input! Try again.")
            else:
                target = env.current_player  # No target needed; self-action

            # Execute the action
            obs, reward, terminated, truncated, info = env.step((card_index, target))
            print(f"You played {env._get_card_name(env.discard_piles[0][-1])}. {info['feedback']} Reward: {reward}")

        else:  # AI's turn
            print("\nAI-1's Turn!")
            card_index = random.randint(0, len(obs['hand']) - 1)
            target = random.choice([i for i in range(env.num_players) if i != env.current_player]) if LoveLetterEnv.CARD_TARGET_REQUIREMENTS[obs['hand'][card_index]] else env.current_player
            obs, reward, terminated, truncated, info = env.step((card_index, target))
            print(f"AI played {env._get_card_name(env.discard_piles[1][-1])}. {info['feedback']} Reward: {reward}")

        # Handle game over
        if terminated or truncated:
            winner = info["winner"]
            print("\nGame Over!")
            print(f"{'You' if winner == 0 else 'AI'} wins!")
            break

if __name__ == "__main__":
    human_vs_random_ai()
