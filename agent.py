import gymnasium as gym
import numpy as np
import pygame

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn 
from torch.utils.tensorboard import SummaryWriter
import yaml 

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse

import flappy_bird_gymnasium
import os

DATE_FORMAT = "%d-%m %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
# DRIVE_FOLDER_PATH = "/content/drive/MyDrive/FlappyBird_DQN"
# os.makedirs(DRIVE_FOLDER_PATH, exist_ok=True)

matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yaml", "r") as f:
            all_hyperparameter_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set
        
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.tau = hyperparameters['tau']

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")
        self.writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, self.hyperparameter_set))


    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as f:
                f.write(log_message + "\n")

        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        clock = pygame.time.Clock() if render else None

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        start_episode = 0
        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            if os.path.exists(self.MODEL_FILE):
                print("Found existing checkpoint. Resuming training...")
                checkpoint = torch.load(self.MODEL_FILE)
                policy_dqn.load_state_dict(checkpoint['model_state_dict'])
                target_dqn.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint['episode'] + 1
                epsilon = checkpoint['epsilon']
                print(f"Resuming from episode {start_episode}.")

            epsilon_history = []

            step_count = 0

            best_reward = -9999999
        else:
            checkpoint = torch.load(self.MODEL_FILE)
            policy_dqn.load_state_dict(checkpoint['model_state_dict'])
            policy_dqn.eval()
            # policy_dqn.load_state_dict(torch.load(self.MODEL_FILE)) 
            # policy_dqn.eval()
        
        initial_state, _ = env.reset()
        sample_input = torch.tensor(initial_state, dtype=torch.float, device=device).unsqueeze(dim=0)
        self.writer.add_graph(policy_dqn, sample_input)
        print("Model graph saved.")

        running = True
        episode = start_episode
        while running:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                
                if not running:
                    break

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1
                
                state = new_state

                if render and clock:
                    clock.tick(60)

            if not running:
                break
            
            rewards_per_episode.append(episode_reward)

            if is_training:
                self.writer.add_scalar('Reward per Episode', episode_reward, episode)
                self.writer.add_scalar('Epsilon', epsilon, episode)
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + "\n")
                    
                    checkpoint = {
                        'episode': episode,
                        'model_state_dict': policy_dqn.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epsilon': epsilon
                    }
                    torch.save(checkpoint, self.MODEL_FILE)
                    best_reward = episode_reward
                
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn, episode)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

            episode += 1
        env.close()
        print("Program ended.")
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        plt.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def soft_update(self, policy_dqn, target_dqn):
        for target_param, policy_param in zip(target_dqn.parameters(), policy_dqn.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def optimize(self, mini_batch, policy_dqn, target_dqn, episode=0):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(policy_dqn, target_dqn)
        self.writer.add_scalar('Loss', loss.item(), episode)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, render=False)
    else:
        dql.run(is_training=False, render=True)