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
            spaces.Discrete(2+1, start=-1),  # Card index to play
            spaces.Discrete(self.num_players+1, start=-1),  # Target player
            spaces.Discrete(8+1, start=0) # Guess for guard     
        ))
        self.observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([8, 8]),  # Two cards in hand
            "public_state": spaces.Box(low=0, high=1, shape=(self.num_players,), dtype=np.int32),
            "discard_piles": spaces.Box(low=0, high=8, shape=(self.num_players, 7), dtype=np.int32)  # Discard piles for both players
        })

        # Initialize game state
        self.reset()

    def reset(self):
        # Initialize the deck
        self.deck = [1] * 5 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 2 + [6] * 1 + [7] * 1 + [8] * 1
        random.shuffle(self.deck)

        # Round plus one everything human turn
        self.round = -1

        # Each player starts with one card
        self.hands = [[self.deck.pop()] for _ in range(self.num_players)]

        # Initialize discard piles and protection statuses
        self.discard_piles = np.zeros((self.num_players, 7), dtype=np.int32)
        self.protected = [False] * self.num_players
        self.current_player = -1
        self.winner = None  # Track the winner

        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current player's observation."""
        return {
            "hand": np.array(self.hands[self.current_player], dtype=np.int32),
            "public_state": np.array(self.protected, dtype=np.int32),
            "discard_piles": np.array(self.discard_piles, dtype=np.int32),
        }

    def _get_card_name(self, card_id):
        """Get the card name from the card ID."""
        card_details = {
            0:"Empty", 1: "Guard", 2: "Priest", 3: "Baron", 4: "Handmaid",
            5: "Prince", 6: "King", 7: "Countess", 8: "Princess"
        }
        return card_details.get(card_id, "Unknown")

    def prepare_turn(self):
        """Shift player, draw a card at the beginning of the turn, and change protection status"""
        # Advance to the next player
        self.current_player = (self.current_player + 1) % self.num_players

        if self.deck:
            self.hands[self.current_player].append(self.deck.pop())

        self.protected[self.current_player] = False
        
        return self._get_observation()
    
    def _action_valid(self, card_index, target, guess):
        """
        validate a player's action to ensure it conforms to game rules.
        :param action: Tuple (card_index, target, guess)
        :return: Boolean
        """
        hand = self.hands[self.current_player]
        card = hand[card_index]

        valid = True

        # 1. Only Guard (card 1) selects Guess (others must have guess = 0)
        if (card == 1 and guess == 0) or (card != 1 and guess != 0):
            valid = False

        # 2. Only Prince (5) can target themselves (others must have target != self.current_player)
        if card != 5 and target == self.current_player:
            valid = False

        # 3. Handmaid (4), Countess (7), and Princess (8) don't have targets (target must be -1)
        if card in [4, 7, 8] and target != -1:
            valid = False

        # 4. Must play Countess if you have Prince and King in hand
        if 7 in hand and (5 in hand or 6 in hand) and card != 7:
            valid = False

        return valid

    def step(self, action):
        card_index, target, guess = action

        if self.current_player == 0:  # Human's turn
            self.round += 1
            card_index, target, guess = action
        else:  # AI's turn
            # Run the opponent's turn using `_run_opp`
            card_index, target, guess = self._run_opp()

        if not self._action_valid(card_index, target, guess):
            reward = -100
            return self._get_observation(), reward, True, False, {"feedback": "Player eliminated by invalid move", "winner": 1 if self.current_player == 0 else 0}

        # Apply the action
        reward, feedback, terminated = self._apply_action(card_index, target, guess)

        # If the game is terminated immediately, return the result
        if terminated:
            self.winner = 0 if len(self.hands[0]) > 0 else 1  # Determine the winner
            return self._get_observation(), reward, terminated, False, {"feedback": feedback, "winner": self.winner}

        # Check for truncation (deck runs out)
        truncated = len(self.deck) == 0
        if truncated:
            self.winner = self._determine_truncated_winner()
            return self._get_observation(), reward, False, truncated, {"feedback": feedback, "winner": self.winner}
        
        # Return updated observation, reward, and status flags
        return self._get_observation(), reward, False, False, {"feedback": feedback}

    def _apply_action(self, card_index, target=None, guess=None):
        card = self.hands[self.current_player].pop(card_index)
        self.discard_piles[self.current_player, self.round] = card

        """Apply the effect of the played card."""
        feedback = ""
        terminated = False

        if card == 1:  # Guard
            guessed_card = guess
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
                    feedback = f"Player is eliminated by Baron. Player's card ({self._get_card_name(eliminated_card)}) was weaker."
                terminated = self._is_game_terminated()  # Check termination immediately
            else:
                feedback = f"Player {target} is protected by Handmaid."

        elif card == 4:  # Handmaid
            self.protected[self.current_player] = True
            feedback = f"Player {self.current_player} are protected from all effects until your next turn."

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

    def _determine_truncated_winner(self):
        """Determine the winner when the game ends due to truncation."""
        human_card = self.hands[0][0] if self.hands[0] else -1
        ai_card = self.hands[1][0] if self.hands[1] else -1
        return 0 if human_card > ai_card else 1  # Compare card values

    def _is_game_terminated(self):
        """Check if the game is over."""
        active_players = [hand for hand in self.hands if len(hand) > 0]
        return len(active_players) <= 1

    def _run_opp(self):
        """Run opponent's turn"""
        # Determine action
        hand = self.hands[self.current_player]
        card_index = random.randint(0, 1)
        # Always target the opponent unless they are protected
        target = -1
        if self.CARD_TARGET_REQUIREMENTS[hand[card_index]]:
            if hand[card_index] == 5 and self.protected[0]:
                target = self.current_player
            else:
                target = 0
        # Guess randomly from the remaining cards (TODO: could be changed to guess largest of most frequent)
        guess = 0
        if hand[card_index] == 1:
            filtered_list = [x for x in self.deck if x != 1]
            if filtered_list:
                guess = random.choice(filtered_list)
            else:
                guess = 2
        
        return (card_index, target, guess)
        

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

def format_discard_piles(env, discard_piles):
    """Format discard piles with card names for display."""
    formatted_piles = []
    for player, pile in enumerate(discard_piles):
        pile_names = [env._get_card_name(card) for card in pile]
        formatted_piles.append(f"Player {player}'s discard pile: {', '.join(pile_names)}")
    return '\n'.join(formatted_piles)

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

        #TODO In training, this should be included in the step function. Here, it is required to get the updated hand info
        obs = env.prepare_turn()

        if env.current_player == 0:  # Human player's turn
            print("\nYour Turn!")
            print(f"Your hand: {[f'{i}: {env._get_card_name(c)}' for i, c in enumerate(obs['hand'])]}")
            print(f"Public state (protection status): {['Protected' if p else 'Unprotected' for p in obs['public_state']]}")
            print(format_discard_piles(env, obs['discard_piles']))

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

            target = -1
            if LoveLetterEnv.CARD_TARGET_REQUIREMENTS[selected_card]:
                while True:
                    try:
                        target = int(input(f"Choose a target player (0: You, 1: AI-1): "))
                        if 0 <= target < env.num_players:
                            break
                    except ValueError:
                        print("Invalid input! Try again.")
            else:
                target = -1 # No target needed; self-action

            guess = 0
            if selected_card==1: # 1 is the Guard card
                while True:
                    try:
                        guess = int(input("Guess the target player's card (1-8): "))
                        if 1 <= guess <= 8:
                            break
                        else:
                            print("Invalid guess! Please enter a number between 1 and 8.")
                    except ValueError:
                        print("Invalid input! Please enter a valid number.")

            # Execute the action
            obs, reward, terminated, truncated, info = env.step((card_index, target, guess))
            print(f"Your remaining hand: {[f'{i}: {env._get_card_name(c)}' for i, c in enumerate(obs['hand'])]}")
        else:  # AI's turn
            print("\nOpponent's Turn!")
            print(f"Public state (protection status): {['Protected' if p else 'Unprotected' for p in obs['public_state']]}")
            print(f"Discard pile: {format_discard_piles(env, obs['discard_piles'])}")
            obs, reward, terminated, truncated, info = env.step((-1, -1, 0))  # AI logic handled in step

        # Display feedback
        print(info["feedback"])
        if "winner" in info:
            print("\nGame Over!")
            print(f"{'You' if info['winner'] == 0 else 'AI'} wins!")
            break

if __name__ == "__main__":
    human_vs_random_ai()