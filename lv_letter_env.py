import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import Counter
from sb3_contrib import TRPO, RecurrentPPO

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

        # self.ai_policy = TRPO.load("trpo_love_letter_4env_1000000")
        self.ai_policy = RecurrentPPO.load("recurrent_ppo_love_letter_1000000")
        self.num_players = 4
        self.num_cards = 16
        self.prev_actions = [[0,0,0,0,0]] * self.num_players
        self.hidden_action_space = spaces.Tuple((
            spaces.Discrete(2, start=0),  # Card index to play
            spaces.Discrete(self.num_players+1, start=-1),  # Target player
            spaces.Discrete(8+1, start=0) # Guess for guard     
        ))
        # Convert to MultiDiscrete
        low = [space.start for space in self.hidden_action_space.spaces]  # Lower bounds
        high = [space.start + space.n - 1 for space in self.hidden_action_space.spaces]  # Upper bounds

        # Flatten into a MultiDiscrete space
        self.action_space = spaces.MultiDiscrete([high[i] - low[i] + 1 for i in range(len(low))])

        self.observation_space = spaces.Dict({
            "round": spaces.Discrete(16+1, start=0),
            "hand":  spaces.Box(low=np.array([1, 1]), high=np.array([8, 8]), dtype=np.int32),
            "public_state": spaces.Box(low=0, high=1, shape=(self.num_players,), dtype=np.int32),
            "active_players": spaces.MultiBinary(self.num_players),
            "discard_piles": spaces.Box(low=0, high=8, shape=(self.num_players, 16), dtype=np.int32),  # Discard piles for both players
            "prev_round_actions": spaces.Box(low=np.array([1, 1, -1, 0, 0] * self.num_players), high=np.array([8, 8, self.num_players, 8, 8] * self.num_players), dtype=np.int32) # played card, unplayed card, target, guess, priest info
        })


        # Initialize game state
        self.reset()

    def reset(self, seed=184, options=None):
        # print("resetting")
        # Initialize the deck
        self.deck = [1] * 5 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 2 + [6] * 1 + [7] * 1 + [8] * 1
        # random.seed(seed)
        random.shuffle(self.deck)
        self.facedown_card = self.deck.pop()

        # Round plus one everything human turn
        self.round = 0

        # Each player starts with one card
        self.hands = [[self.deck.pop()] for _ in range(self.num_players)]
        self.hands[0].append(self.deck.pop())

        # Initialize discard piles and protection statuses
        # self.discard_piles = np.zeros((self.num_players, 7), dtype=np.int32)
        self.discard_piles = [[] for _ in range(self.num_players)]
        self.protected = [False] * self.num_players
        self.active = [True] * self.num_players
        self.current_player = 0
        self.winner = None  # Track the winner
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current player's observation."""
        padded_discard_piles = [
            self.discard_piles[player] + [0] * (16 - len(self.discard_piles[player]))
            for player in range(self.num_players)
        ]
        hist = [x.copy() for x in self.prev_actions]
        for i in range(self.num_players):
            if i != self.current_player:
                hist[i][1] = 0
                hist[i][4] = 0
        return {
            "round": self.round,
            "hand": np.array([x for x in self.hands[self.current_player]], dtype=np.int32),
            "public_state": np.array(self.protected, dtype=np.int32),
            "active_players": np.array(self.active, dtype=np.int32),
            "discard_piles": np.array(padded_discard_piles, dtype=np.int32),
            "prev_round_actions": np.array([item for sublist in hist for item in sublist]),
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
        # print("-----")
        # print(self.current_player)
        # print(self._get_observation())
        # print(card_index)
        # print(target)
        # print(self.deck)
        card = hand[card_index]

        valid = True
        # print(f"{card} {target} {guess}")

        # 1. Only Guard (card 1) selects Guess (others must have guess = 0)
        if (card == 1 and (guess == 0 or guess == 1)) or (card != 1 and guess != 0):
            # print("1")
            valid = False

        # 2. Only Prince (5) can target themselves (others must have target != self.current_player)
        if card != 5 and target == self.current_player:
            # print("2")
            valid = False

        # 3. Handmaid (4), Countess (7), and Princess (8) don't have targets (target must be -1)
        if card in [4, 7, 8] and target != -1:
            # print("3")
            valid = False

        # 4. Must play Countess if you have Prince and King in hand
        if 7 in hand and (5 in hand or 6 in hand) and card != 7:
            # print("4")
            valid = False
            
        # 5. Must target an active player    
        if target > -1 and self.active[target] == False:
            # print("5")
            valid = False

        # 6. Cannot target a protected player unless all are protected
        if all(not active or protected for idx, (active, protected) in enumerate(zip(self.active, self.protected)) if idx != self.current_player):
            if card == 5 and target != self.current_player:
                # print("6")
                valid = False
        elif target > -1 and self.protected[target] == True:
            # print("7")
            valid = False
            
        # 7. Must target a player if card requires a target
        if self.CARD_TARGET_REQUIREMENTS[hand[card_index]]:
            if target == -1:
                valid = False

        return valid

    def step(self, action):
        # Do own turn
        obs, reward, terminated, truncated, info = self.substep(action)
        if self.active[0] == 0:
            # print("died")
            terminated = 1
        if terminated or truncated:
            return obs, reward, True, False, info
        
        # Do opponents' turns
        for i in range(1, self.num_players):
            if self.active[(self.current_player + 1) % self.num_players] == False:
                if not any(self.active):
                    print("error")
                    break
                self.current_player = (self.current_player + 1) % self.num_players
                continue
            self.prepare_turn()
            obs, reward, terminated, truncated, info = self.substep(action)
            if self.active[0] == 0:
                # print("died")
                terminated = 1
            if terminated or truncated:
                return obs, reward, True, False, info
            
        self.prepare_turn()

        obs = self._get_observation()

        return obs, reward, False, False, {}


    def substep(self, action):

        hand = self.hands[self.current_player]

        if self.current_player == 0:  # Human's turn
            self.round += 1
            card_index, target, guess = action
            target -= 1
        else:  # AI's turn
            # Run the opponent's turn using `_run_opp`
            if len(self.hands[self.current_player]) != 2:
                print("ERROR")
            # card_index, target, guess = self._run_opp()
            card_index, target, guess = self._run_opp_TRPO()
            # card_index, target, guess = self._run_opp_random()
            
        hand = self.hands[self.current_player]
        self.prev_actions[self.current_player] = [hand[card_index], hand[(card_index + 1)%2], target, guess, 0]


        if not self._action_valid(card_index, target, guess):
            reward = -100
            #TODO put both cards in discard pile
            card = self.hands[self.current_player].pop(card_index)
            # self.discard_piles[self.current_player, self.round] = card
            self.discard_piles[self.current_player].append(card)
            self.hands[self.current_player] = []
            self.active[self.current_player] = False
            terminated = self._is_game_terminated()
            truncated = len(self.deck) == 0
            if terminated:
                self.winner = self._determine_terminated_winner()
                # print("Terminated")
                return self._get_observation(), reward, terminated, False, {"feedback": "Player eliminated by invalid move", "winner": self.winner}
            elif truncated:
                # print("Truncated")
                self.winner = self._determine_truncated_winner()
                return self._get_observation(), reward, False, truncated, {"feedback": "Game ended by truncation", "winner": self.winner}
            else:
                return self._get_observation(), reward, terminated, False, {"feedback": "Player eliminated by invalid move"}

        # Apply the action
        reward, feedback, terminated = self._apply_action(card_index, target, guess)

        # If the game is terminated immediately, return the result
        if terminated:
            # print("Terminated")
            self.winner = self._determine_terminated_winner()
            reward = 100 if self.winner == 0 else -100
            return self._get_observation(), reward, terminated, False, {"feedback": feedback, "winner": self.winner}

        # Check for truncation (deck runs out)
        truncated = len(self.deck) == 0
        if truncated:
            # print("Truncated")
            self.winner = self._determine_truncated_winner()
            reward = 100 if self.winner == 0 else -100
            return self._get_observation(), reward, False, truncated, {"feedback": feedback, "winner": self.winner}
        
        # Return updated observation, reward, and status flags
        return self._get_observation(), reward, False, False, {"feedback": feedback}

    def _apply_action(self, card_index, target=None, guess=None):
        card = self.hands[self.current_player].pop(card_index)
        # self.discard_piles[self.current_player, self.round] = card
        self.discard_piles[self.current_player].append(card)

        """Apply the effect of the played card."""
        reward = 0
        feedback = ""
        terminated = False

        if card == 1:  # Guard
            guessed_card = guess
            if not self.protected[target] and guessed_card in self.hands[target]:
                # self.discard_piles[target, self.round] = guessed_card
                self.discard_piles[target].append(guessed_card)
                self.hands[target].remove(guessed_card)  # Target is eliminated
                self.active[target] = False
                reward = 10
                feedback = f"Player {target} is eliminated because their card ({self._get_card_name(guessed_card)}) was guessed."
                terminated = self._is_game_terminated()  # Check termination immediately
            else:
                feedback = f"The guess ({self._get_card_name(guessed_card)}) was incorrect."

        elif card == 2:  # Priest
            if not self.protected[target]:
                revealed_card = self.hands[target][0] if self.hands[target] else None
                self.prev_actions[self.current_player][4] = revealed_card
                feedback = f"Player {target}'s hand is revealed: {self._get_card_name(revealed_card)}."
            else:
                feedback = f"Player {target} is protected by Handmaid."

        elif card == 3:  # Baron
            if not self.protected[target]:
                if self.hands[self.current_player][0] > self.hands[target][0]:
                    self.active[target] = False
                    eliminated_card = self.hands[target].pop(0)  # Target is eliminated
                    # self.discard_piles[target, self.round] = eliminated_card
                    self.discard_piles[target].append(eliminated_card)
                    reward = 10
                    feedback = f"Player {target} is eliminated by Baron. Their card ({self._get_card_name(eliminated_card)}) was weaker."

                else:
                    self.active[self.current_player] = False
                    eliminated_card = self.hands[self.current_player].pop(0)  # Current player is eliminated
                    # self.discard_piles[self.current_player, self.round] = eliminated_card
                    self.discard_piles[self.current_player].append(eliminated_card)
                    reward = -100
                    feedback = f"Current player is eliminated by Baron. Player's card ({self._get_card_name(eliminated_card)}) was weaker."
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
                    self.active[target] = False
                    reward = 10
                    feedback = f"Player {target} discarded the Princess and is eliminated!"
                    # self.discard_piles[target, self.round] = 8
                    self.discard_piles[target].append(8)
                    self.hands[target] = []  # Eliminate the player
                    terminated = self._is_game_terminated()  # Check termination immediately
                else:
                    new_card = self.deck.pop() if self.deck else self.facedown_card
                    # self.discard_piles[target, self.round] = discarded_card
                    self.discard_piles[target].append(discarded_card)
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
            self.active[self.current_player] = False
            # self.discard_piles[self.current_player, self.round] = 8
            self.discard_piles[self.current_player].append(8)
            self.hands[self.current_player] = []  # Current player eliminated
            reward = -100
            feedback = "You are eliminated for discarding the Princess."
            terminated = self._is_game_terminated()  # Check termination immediately

        return reward, feedback, terminated

    def _determine_truncated_winner(self):
        """Determine the winner when the game ends due to truncation."""
        final_cards = [x[0] if x else -1 for x in self.hands]
        # print(f"Final cards: {final_cards}")
        return np.argmax(final_cards) # Compare card values
    
    def _determine_terminated_winner(self):
        """Determine the winner when the game ends due to termination."""
        indices = np.where(self.active)[0]
        return indices[0] # Get index of active player

    def _is_game_terminated(self):
        """Check if the game is over."""
        return sum(self.active) == 1

    def _run_opp(self):
        """Run opponent's turn"""
        # Determine action
        hand = self.hands[self.current_player]
        # print(hand)
        if 8 in hand:
            card_index = (hand.index(8) + 1) % 2
        elif 7 in hand:
            if 5 in hand or 6 in hand:
                card_index = hand.index(7)
            else:
                card_index = (hand.index(7) + 1) % 2
        elif 6 in hand:
            card_index = (hand.index(6) + 1) % 2
        elif 3 in hand:
            if 5 not in hand:
                card_index = (hand.index(3) + 1) % 2
            else:
                card_index = random.randint(0, 1)
        else:
            card_index = random.randint(0, 1)
        # Determine target
        target = -1
        # print("-----")
        # print(self._get_observation())
        # print(self.current_player)
        # print(card_index)
        if self.CARD_TARGET_REQUIREMENTS[hand[card_index]]:
            if hand[card_index] == 5:
                alive = [index for index, status in enumerate(self.active) if status]
                unprotected = [index for index, status in enumerate(self.protected) if not status]
                options = [item for item in alive if item in unprotected]
                target = random.choice(options)
            else:
                alive = [index for index, status in enumerate(self.active) if status and index != self.current_player]
                unprotected = [index for index, status in enumerate(self.protected) if not status and index != self.current_player]
                options = [item for item in alive if item in unprotected]
                if options:
                    target = random.choice(options)
                else:
                    target = random.choice(alive)
        # Guess the highest value remaining card from the cards with highest multiplicity left
        guess = 0
        if hand[card_index] == 1:
            filtered_deck = [x for x in self.deck if x != 1]
            if filtered_deck:
                counter = Counter(filtered_deck)
                max_count = max(counter.values())
                most_frequent_cards = [card for card, count in counter.items() if count == max_count]
                guess = max(most_frequent_cards)
            else:
                guess = 2
        
        return (card_index, target, guess)
    
    def _run_opp_random(self):
        """Run opponent's turn"""
        # Determine action
        hand = self.hands[self.current_player]
        # print(hand)

        if 7 in hand:
            if 5 in hand or 6 in hand:
                card_index = hand.index(7)
            else:
                card_index = random.randint(0, 1)
        else:
            card_index = random.randint(0, 1)
        # Determine target
        target = -1
        if self.CARD_TARGET_REQUIREMENTS[hand[card_index]]:
            if hand[card_index] == 5:
                alive = [index for index, status in enumerate(self.active) if status]
                unprotected = [index for index, status in enumerate(self.protected) if not status]
                options = [item for item in alive if item in unprotected]
                target = random.choice(options)
            else:
                alive = [index for index, status in enumerate(self.active) if status and index != self.current_player]
                unprotected = [index for index, status in enumerate(self.protected) if not status and index != self.current_player]
                options = [item for item in alive if item in unprotected]
                if options:
                    target = random.choice(options)
                else:
                    target = random.choice(alive)
        # Guess randomly from the remaining cards
        guess = 0
        if hand[card_index] == 1:
            guess = random.randint(2, 8)

        return (card_index, target, guess)
    
    def _run_opp_TRPO(self):
        obs = self._get_observation()
            
        # Use the trained policy to predict the AI's action
        action, _ = self.ai_policy.predict(obs)
        
        # Decode action into card_index, target, and guess
        card_index, target, guess = action
        return card_index, target-1 , guess

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
        if env.active[(env.current_player + 1) % env.num_players] == False:
            if not any(env.active):
                print("error")
                break
            env.current_player = (env.current_player + 1) % env.num_players
            continue
            
        print("\n" + "-" * 50)  # Separation between turns

        #TODO In training, this should be included in the step function. Here, it is required to get the updated hand info
        obs = env.prepare_turn()

        if env.current_player == 0:  # Human player's turn
            print("\nYour Turn!")
            print(f"Your hand: {[f'{i}: {env._get_card_name(c)}' for i, c in enumerate(obs['hand'])]}")
            print(f"Public state (protection status): {['Protected' if p else 'Unprotected' for p in obs['public_state']]}")
            print(f"Active players: {['Yes' if p else 'No' for p in obs['active_players']]}")
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
                        target = int(input(f"Choose a target player (0: You, {', '.join([f'{i+1}: Opponent-{i+1}' for i in range(env.num_players-1)])})"))
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
            obs, reward, terminated, truncated, info = env.substep((card_index, target, guess))
            print(f"Your remaining hand: {[f'{i}: {env._get_card_name(c)}' for i, c in enumerate(obs['hand'])]}")
        else:  # AI's turn
            print(f"\nOpponent {env.current_player}'s Turn!")
            print(f"Public state (protection status): {['Protected' if p else 'Unprotected' for p in obs['public_state']]}")
            print(f"Active players: {['Yes' if p else 'No' for p in obs['active_players']]}")
            print(f"Discard pile: {format_discard_piles(env, obs['discard_piles'])}")
            obs, reward, terminated, truncated, info = env.substep((-1, -1, 0))  # AI logic handled in step

        # Display feedback
        print(info["feedback"])
        if "winner" in info:
            print("\nGame Over!")
            print(f"Player {info['winner']} wins!")
            break

if __name__ == "__main__":
    human_vs_random_ai()