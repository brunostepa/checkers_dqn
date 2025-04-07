import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from collections import deque

# Define Checkers Environment
class CheckersEnv:
    def __init__(self):
        self.board = self.reset()
    
    def reset(self):
        self.board = np.zeros((8, 8))
        for i in range(3):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.board[i, j] = -1
        for i in range(5, 8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.board[i, j] = 1
        return self.get_state()
    
    def get_state(self):
        return self.board.flatten()
    
    def get_valid_moves(self, player):
        moves = []
        direction = -1 if player == 1 else 1
        for i in range(8):
            for j in range(8):
                if self.board[i, j] == player:
                    if i + direction in range(8):
                        if j > 0 and self.board[i + direction, j - 1] == 0:
                            moves.append((i, j, i + direction, j - 1))
                        if j < 7 and self.board[i + direction, j + 1] == 0:
                            moves.append((i, j, i + direction, j + 1))
                        if i + 2 * direction in range(8):
                            if j > 1 and self.board[i + direction, j - 1] == -player and self.board[i + 2 * direction, j - 2] == 0:
                                moves.append((i, j, i + 2 * direction, j - 2))
                            if j < 6 and self.board[i + direction, j + 1] == -player and self.board[i + 2 * direction, j + 2] == 0:
                                moves.append((i, j, i + 2 * direction, j + 2))
        return moves
    
    def step(self, action, player):
        i, j, ni, nj = action
        reward = 0
        if abs(i - ni) == 2:
            self.board[(i + ni) // 2, (j + nj) // 2] = 0
            reward = 1
        self.board[ni, nj] = player
        self.board[i, j] = 0
        done = not self.get_valid_moves(1) and not self.get_valid_moves(-1)
        return self.get_state(), reward, done
    
    def render(self):
        plt.imshow(self.board, cmap='coolwarm', interpolation='nearest')
        plt.show(block=False)
        plt.pause(0.5)
        plt.clf()

# Define DQN Model
class DQN(nn.Module):
    def __init__(self, state_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim + 4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Train DQN Agent
class DQNAgent:
    def __init__(self, state_dim):
        self.model = DQN(state_dim)
        self.target_model = DQN(state_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_action(self, state, env, player):
        valid_moves = env.get_valid_moves(player)
        if not valid_moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = []
        for move in valid_moves:
            move_tensor = torch.tensor(move, dtype=torch.float32).unsqueeze(0)
            combined_input = torch.cat((state_tensor, move_tensor), dim=1)
            with torch.no_grad():
                q_values.append(self.model(combined_input).item())
        best_action_index = np.argmax(q_values)
        return valid_moves[best_action_index]
    
    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        combined_states = torch.cat((states, actions), dim=1)
        q_values = self.model(combined_states)
        targets = rewards + (1 - dones) * self.gamma * q_values.detach()
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Main Training Loop
env = CheckersEnv()
agent = DQNAgent(state_dim=64)
for episode in range(1000000):
    state = env.reset()
    done = False
    player_turn = 1
    while not done:
        action = agent.select_action(state, env, player_turn)
        if action is None:
            break  
        next_state, reward, done = env.step(action, player_turn)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.train()
        player_turn *= -1
    agent.decay_epsilon()
    agent.update_target_network()
    if episode % 100000 == 0:
        torch.save(agent.model.state_dict(), f"checkers_dqn_test_{episode}.pth")
    print(f"Episode {episode + 1}: Epsilon {agent.epsilon}")
torch.save(agent.model.state_dict(), "checkers_dqn.pth")


def test_agent():
    env = CheckersEnv()
    agent = DQNAgent(state_dim=64)
    agent.model.load_state_dict(torch.load("checkers_dqn.pth"))
    state = env.reset()
    done = False
    player_turn = 1  

    while not done:
        env.render()
        if player_turn == 1:
            valid_moves = env.get_valid_moves(1)
            if not valid_moves:
                print("No valid moves left. AI wins!")
                break
            print("Your valid moves:", valid_moves)
            move = None
            while move not in valid_moves:
                try:
                    move_input = input("Enter your move (row_from col_from row_to col_to): ")
                    move = tuple(map(int, move_input.split()))
                    if move not in valid_moves:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Invalid input! Enter four numbers.")
            state, _, done = env.step(move, 1)
        else:
            action = agent.select_action(state, env, -1)
            if action is None:
                print("AI has no moves left. You win!")
                break
            print(f"AI moves: {action}")
            state, _, done = env.step(action, -1)
        player_turn *= -1
    env.render()
    print("Game over!")

#test_agent()
