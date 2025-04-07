import numpy as np
import pandas as pd
import random
import chess
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
from collections import deque
from IPython.display import display, clear_output

move_history = {}

# Define piece values for reward calculation
piece_value = {
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,  # White pieces
    'P': -1, 'N': -3, 'B': -3, 'R': -5, 'Q': -9, 'K': 0  # Black pieces
}

# Chess Board Representation Functions
def create_rep_layer(board, piece_type):
    board_fen = board.board_fen()  # Get board in FEN notation
    board_fen = board_fen.replace("/", "")  # Remove row separators

    # Mapping pieces to numerical values
    piece_map = {
        piece_type: -1, piece_type.upper(): 1
    }

    # Convert FEN characters into 8x8 matrix
    board_mat = np.zeros((8, 8), dtype=int)
    row, col = 0, 0

    for char in board_fen:
        if char.isdigit():
            col += int(char)  # Empty squares
        else:
            if char in piece_map:
                board_mat[row, col] = piece_map[char]
            col += 1
        if col >= 8:
            row += 1
            col = 0

    return board_mat

def board_2_rep(board):
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    return np.stack([create_rep_layer(board, piece) for piece in pieces])

def move_2_rep(move, board):
    move = board.push_san(move)
    move_str = str(board.pop())
    from_layer, to_layer = np.zeros((8,8)), np.zeros((8,8))
    from_layer[8 - int(move_str[1]), ord(move_str[0]) - ord('a')] = 1
    to_layer[8 - int(move_str[3]), ord(move_str[2]) - ord('a')] = 1
    return np.stack([from_layer, to_layer])

# DQN Agent Class
class DQNAgent:
    def __init__(self, state_shape, action_shape, model=None):
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        if model is None:
            self.model = self.build_model(state_shape, action_shape)
        else:
            self.model = model  # Use the loaded model
        self.target_model = self.build_model(state_shape, action_shape)
        self.target_model.set_weights(self.model.get_weights())
    
    def build_model(self, state_shape, action_shape):
        model = Sequential([
            Flatten(input_shape=state_shape),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(np.prod(action_shape), activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model

    def select_action(self, state, board):
        valid_moves = list(board.legal_moves)
        if not valid_moves:
            return None
        if np.random.rand() <= 0.1:
            return random.choice(valid_moves)
        
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]

        # Fix: Use only `from_square` as index
        best_move = max(valid_moves, key=lambda move: q_values[move.from_square])
        
        return best_move
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        targets = self.model.predict(np.array(states), verbose=0)
        next_q_values = self.target_model.predict(np.array(next_states), verbose=0)

        for i in range(batch_size):
            move = actions[i]  # Get the move object
            move_index = move.from_square  # Use `from_square` directly, which is between 0 and 63

            if dones[i]:
                targets[i, move_index] = rewards[i]
            else:
                targets[i, move_index] = rewards[i] + 0.99 * np.max(next_q_values[i])

        self.model.fit(np.array(states), targets, epochs=1, verbose=0)
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Environment Step Function
def calculate_reward(board, move):
    global move_history
    reward = 0
    move_str = board.san(move)

    # Material difference before and after the move
    initial_material = sum(piece_value[p.symbol()] for p in board.piece_map().values())
    board.push(move)
    final_material = sum(piece_value[p.symbol()] for p in board.piece_map().values())
    material_gain = final_material - initial_material
    reward += material_gain * 10  # Scale material gains
    if not board.turn:  # If it's Black's turn (after White moved), flip reward
        reward = -reward 
    # Encourage mobility
    reward += len(list(board.legal_moves)) * 0.1

    # Positional bonuses
    move_to = move.to_square
    row, col = divmod(move_to, 8)
    if (row, col) in [(3, 3), (3, 4), (4, 3), (4, 4)]:  # Center control
        reward += 0.5
    if move.promotion:  # Reward pawn promotion
        reward += 8
    
    # Tactical awareness
    if board.is_check():
        reward += 1  # Bonus for putting opponent in check
    if board.is_checkmate():
        reward += 100  # Huge reward for checkmate
    if board.is_stalemate():
        reward -= 10  # Penalize stalemates

    # Penalize bad moves
    if board.is_insufficient_material():
        reward -= 5  # Prevent pointless moves

    if move_str in move_history:
        move_history[move_str] += 1
        reward -= move_history[move_str] * 5  
    else:
        move_history[move_str] = 1

    board.pop() 
    return reward


def take_action(board, action):
    reward = calculate_reward(board, action)

    board.push(action)  # Make the move
    
    # Reward is the material difference
    #reward = sum([piece_value[piece.symbol()] for piece in board.piece_map().values()])

    # Check if the game is over
    done = board.is_game_over()

    # Get the next state
    next_state = board_2_rep(board)

    return next_state, reward, done

# Training Loop
def train_dqn(agent, num_episodes=100):
    global move_history
    with open("chess_rewards.log", "a") as f:
        for episode in range(num_episodes):
            board = chess.Board()
            state = board_2_rep(board)
            done, episode_reward = False, 0
            move_history = {}
            while not done:
                action = agent.select_action(state, board)
                if action is None:
                    done = True  # Ensure exit if no legal moves
                    continue
                next_state, reward, done = take_action(board, action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                episode_reward += reward
            agent.update_target_model()
            agent.decay_epsilon()
            f.write(f"Episode {episode+1}, Reward: {episode_reward}\n")
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}")
            if (episode + 1) % 10000 == 0:
                agent.model.save(f"chess_dqn_model_{episode+1}.keras")
        #agent.model.save("chess_dqn_model.h5")

def play_game(agent):
    # Start a new game
    board = chess.Board()
    
    # Display the board initially
    display(board)
    
    # Loop until the game is over
    while not board.is_game_over():
        # Human move
        print("\nYour turn! (Enter your move in SAN, e.g., 'e2e4'):")
        human_move = input().strip()

        try:
            # Try to push the human move
            move = board.push_san(human_move)
        except ValueError:
            print("Invalid move. Try again.")
            continue
        
        # Display the board after the human move
        print("\nBoard after your move:")
        display(board)

        if board.is_game_over():
            break
        
        # Now let the agent (model) make its move
        # Convert the current board state to its numerical representation
        state = board_2_rep(board)
        
        # Let the agent select its move
        model_move = agent.select_action(state, board)
        
        if model_move is None:
            print("No legal moves left for the agent.")
            break
        
        # Push the move to the board
        board.push(model_move)

        # Display the board after the model's move
        print("\nBoard after the model's move:")
        display(board)

    # Print the result of the game
    print("\nGame over!")
    result = board.result()
    print("Result: " + result)

# Load the trained model
loaded_model = load_model("chess_dqn_model_10000_minus.keras") 

# Initialize and Train
state_shape, action_shape = (6, 8, 8), (64,)
#agent = DQNAgent(state_shape, action_shape)
agent = DQNAgent(state_shape, action_shape, model=loaded_model)

#train_dqn(agent, num_episodes=1000000)
play_game(agent)
