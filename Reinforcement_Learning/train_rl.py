# train_rl.py
import torch
import torch.optim as optim
import torch.nn as nn
import random
from collections import deque
import numpy as np

from model import DQN
from simple_env import reset_environment, step_environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 8
output_dim = 2

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
memory = deque(maxlen=50000)
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay = 5000
steps_done = 0
target_update = 1000

def select_action(state, epsilon):
    if random.random() < epsilon:
        return np.random.uniform(low=[-50, 0], high=[50, 100], size=(2,))
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state)
            return action.cpu().numpy()[0]

def compute_epsilon(step):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * step / epsilon_decay)

for episode in range(1000):
    state = reset_environment()
    total_reward = 0

    for t in range(300):
        epsilon = compute_epsilon(steps_done)
        action = select_action(state, epsilon)

        next_state, reward, done = step_environment(state, action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        steps_done += 1

        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

            q_values = policy_net(states)
            next_q_values = target_net(next_states)

            target_q = rewards + gamma * (1 - dones) * next_q_values.max(1)[0].unsqueeze(1)

            loss = nn.MSELoss()(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if steps_done % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    print(f"Episode {episode}, Reward {total_reward}")

    if episode % 50 == 0:
        torch.save(policy_net.state_dict(), f"saved_models/dqn_model_{episode}.pth")
