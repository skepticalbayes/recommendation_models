import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import pickle

device = torch.device('cuda')
cuda = torch.device('cuda')
frame_size = 10


ratings = pickle.load(open('../data/ratings_pos_11.pkl', 'rb'))
movies = torch.load('../data/enc_emb.pt')
movies = dict([i, u] for i, u in enumerate(movies))
users = list(ratings.keys())
id_to_index = dict([(u, i) for i, u in enumerate(pd.read_csv('../data/ml-20m/movies.csv')['movieId'].values)])


class ML20mDataset(Dataset):
    def __init__(self):
        self.set_dataset(1)

    def set_dataset(self, u):
        self.user = u
        self.dataset = ratings[u]

    def __len__(self):
        return max(len(self.dataset) - frame_size, 0)

    def __getitem__(self, idx):
        ratings = self.dataset[idx:frame_size + idx + 1]
        movie_chosen = ratings[:, 0][-1]
        films_watched = ratings[:, 0][:-1]

        films_lookup = torch.stack([movies[id_to_index[i]] for i in ratings[:, 0]])

        state = films_lookup[:-1].to(cuda).float()
        next_state = films_lookup[1:].to(cuda).float()

        rewards = torch.tensor(ratings[:, 1][:frame_size]).to(cuda).float()
        next_rewards = torch.tensor(ratings[:, 1][1:frame_size + 1]).to(cuda).float()

        action = films_lookup[-1].to(cuda)

        reward = torch.tensor(ratings[:, 1][-1].tolist()).to(cuda).float()
        done = torch.tensor(idx == self.__len__() - 1).to(cuda).float()

        state = (state, rewards)
        next_state = (next_state, next_rewards)

        return state, action, reward, next_state, done


dset = ML20mDataset()


class StateRepresentation(nn.Module):
    def __init__(self, frame_size):
        super(StateRepresentation, self).__init__()
        self.frame_size = frame_size
        self.state_lin = nn.Sequential(
            # 33 = embed_size + rating
            nn.Linear(self.frame_size * 33, 32),
            nn.Tanh()
        ).to(cuda)

    def forward(self, info, rewards):
        rewards = torch.unsqueeze(rewards, 2)
        state = torch.cat([info, rewards], 2)
        state = state.view(state.size(0), -1)
        state = self.state_lin(state)
        return state


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, frame_size, init_w=3e-3):
        super(Actor, self).__init__()

        self.frame_size = frame_size
        self.state_rep = StateRepresentation(frame_size)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, info, rewards):
        state = self.state_rep(info, rewards)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return state, x

    def get_action(self, info, rewards):
        state, action = self.forward(info, rewards)
        return state, action


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        action = torch.squeeze(action)
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class GaussianExploration(object):
    def __init__(self, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):
        self.low = -1
        self.high = 1
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def get_action(self, action, t=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = action + np.random.normal(size=len(action)) * sigma
        return np.clip(action, self.low, self.high)


def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def td3_update(step,
               batch,
               gamma=0.99,
               soft_tau=1e-2,
               noise_std=0.2,
               noise_clip=0.5,
               policy_update=2,
               ):
    state, action, reward, next_state, done = batch

    reward = reward.unsqueeze(1)
    done = done.unsqueeze(1)

    enc_state = target_policy_net.state_rep(*state)
    enc_next_state, next_action = target_policy_net(*next_state)
    noise = torch.normal(torch.zeros(next_action.size()), noise_std).to(device)
    noise = torch.clamp(noise, -noise_clip, noise_clip)
    next_action += noise

    target_q_value1 = target_value_net1(enc_next_state, next_action)
    target_q_value2 = target_value_net2(enc_next_state, next_action)
    target_q_value = torch.min(target_q_value1, target_q_value2)
    expected_q_value = reward + (1.0 - done) * gamma * target_q_value

    q_value1 = value_net1(enc_state, action)
    q_value2 = value_net2(enc_state, action)

    value_loss1 = value_criterion(q_value1, expected_q_value.detach())
    value_loss2 = value_criterion(q_value2, expected_q_value.detach())

    value_optimizer1.zero_grad()
    value_loss1.backward(retain_graph=True)
    value_optimizer1.step()

    value_optimizer2.zero_grad()
    value_loss2.backward(retain_graph=True)
    value_optimizer2.step()

    if step % policy_update == 0:
        policy_loss = value_net1(enc_state, policy_net(*state)[1])
        policy_loss = -policy_loss.mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        soft_update(value_net1, target_value_net1, soft_tau=soft_tau)
        soft_update(value_net2, target_value_net2, soft_tau=soft_tau)
        soft_update(policy_net, target_policy_net, soft_tau=soft_tau)

        return (value_loss1.item() + value_loss2.item()) / 2, policy_loss.item()
    return False, False


noise = GaussianExploration(32)

value_net1 = Critic(32, 32, 64).to(cuda)
value_net2 = Critic(32, 32, 64).to(cuda)
policy_net = Actor(32, 32, 64, 10).to(cuda)

target_value_net1 = Critic(32, 32, 64).to(cuda)
target_value_net2 = Critic(32, 32, 64).to(cuda)
target_policy_net = Actor(32, 32, 64, 10).to(cuda)

soft_update(value_net1, target_value_net1, soft_tau=1.0)
soft_update(value_net2, target_value_net2, soft_tau=1.0)
soft_update(policy_net, target_policy_net, soft_tau=1.0)

value_criterion = nn.MSELoss()

policy_lr = 1e-4
value_lr = 1e-5

value_optimizer1 = optim.Adam(value_net1.parameters(), lr=value_lr)
value_optimizer2 = optim.Adam(value_net2.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


v_loss, p_loss = [], []

b_size = 100
step = 0


def form_batch(batch_list):
    b = []
    for i in batch_list:
        if isinstance(i[0], tuple):
            b.append((
                torch.stack([t[0] for t in i]).to(cuda),
                torch.stack([t[1] for t in i]).to(cuda)
            ))
        else:
            b.append(torch.stack(i).to(cuda))
    return b


current_batch = [[] for i in range(5)]

for epoch in range(1):
    for u in tqdm(users[:10000]):
        dset.set_dataset(u)
        for b in range(len(dset)):
            if np.random.rand() > 0.2:  # intake percents
                continue
            minibatch = dset[b]
            [current_batch[i].append(minibatch[i]) for i in range(5)]
            if len(current_batch[1]) >= b_size:
                current_batch = form_batch(current_batch)
                value_loss, policy_loss = td3_update(step, current_batch)
                if value_loss:
                    v_loss.append(value_loss)
                    p_loss.append(policy_loss)
                step += 1
                current_batch = [[] for i in range(5)]


torch.save(target_policy_net.state_dict(), "../models/target_policy.pt")
torch.save(target_value_net1.state_dict(), "../models/target_value1.pt")
torch.save(target_value_net2.state_dict(), "../models/target_value2.pt")
