import random
from collections import deque,namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# 经验回放Class
class experience_replay:
    def __init__(self,replay_buffer_size,batch_size):
        self.replay_buffer = deque()
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

    def replay_buffer_append(self,state,action,next_state,reward):
        if len(self.replay_buffer)<self.replay_buffer_size:
            self.replay_buffer.append(Transition(state,action,next_state,reward))
        if len(self.replay_buffer)>self.replay_buffer_size:
            self.replay_buffer.popleft()

    def batch_sample(self):
        return random.sample(self.replay_buffer,self.batch_size)

    def len(self):
        return len(self.replay_buffer)

from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
# DQN网络Class
class DQN(nn.Module):
    def __init__(self,n_states,n_actions):
        super(DQN,self).__init__()
        self.DQN_NetWork = nn.Sequential(
            nn.Linear(n_states,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,n_actions)
        )

    def forward(self,x):
        return self.DQN_NetWork(x)

# 智能体Class
class Agent:
    def __init__(self, n_states, n_actions, eta=0.5, gamma=0.99, replay_buffer_size=10000, batch_size=32):
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = eta
        self.gamma = gamma
        self.replay_buffer = experience_replay(replay_buffer_size,batch_size)
        self.batch_size = batch_size
        self.model = DQN(n_states,n_actions)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.0001)


    def _replay(self):
        if self.replay_buffer.len() < self.batch_size:
            return

        batch = self.replay_buffer.batch_sample()
        batch = Transition(*zip(*batch))
        # 32 * 4
        state_batch = torch.cat(batch.state)

        # 32 * 1
        action_batch = torch.cat(batch.action)

        # 32
        reward_batch = torch.cat(batch.reward)

        # <32 * 4
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        #Eval Mode
        self.model.eval()
        # pred Qsa
        # 32 * 2
        state_action_value = self.model(state_batch).gather(dim = 1,index = action_batch)


        # True Qsa
        non_none_musk = torch.ByteTensor(tuple(map(lambda s : s is not None,batch.next_state)))

        # 32
        next_state_action_values = torch.zeros(self.batch_size)
        # 32
        next_state_action_values[non_none_musk] = self.model(next_state_batch).max(dim = 1)[0].detach()

        true_state_action_values = reward_batch + self.gamma * next_state_action_values

        # Train Mode
        self.model.train()

        loss = F.smooth_l1_loss(state_action_value.squeeze(),true_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_Q_netWork(self):
        self._replay()

    def memorize(self,state,action,next_state,reward):
        self.replay_buffer.replay_buffer_append(state,action,next_state,reward)

    def epsilon_Greedy(self,state,epsilon = 0.1):
        best_index = np.array(self.model(state).max(1)[1])[0]
        prob = np.ones(self.n_actions)*epsilon/self.n_actions
        prob[best_index] = prob[best_index] + (1 - epsilon)
        action = np.random.choice(self.n_actions,p = prob)
        action = torch.tensor([[action]])
        return action

import gym
env = gym.make('CartPole-v0')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
complete_episodes = 0
max_episodes = 500
max_steps = 500
agent = Agent(n_state,n_action)

for episodes in range(max_episodes):
    state = env.reset()
    state = torch.from_numpy(state).unsqueeze(0)
    for steps in range(max_steps):
        #env.render()
        action = agent.epsilon_Greedy(state)
        next_state,_,done,_ = env.step(action.item())
        if done:
            next_state = None

            if steps < 195:
                reward = torch.FloatTensor([-1.])
                complete_episodes = 0
            else:
                reward = torch.FloatTensor([1])
                complete_episodes += 1
        else:
            reward = torch.FloatTensor([0])
            next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
            next_state = next_state.unsqueeze(0)
        agent.memorize(state, action, next_state, reward)
        agent.update_Q_netWork()
        state = next_state

        if done:
            print(f'episode: {episodes}, steps: {steps}')
            break

    if complete_episodes >= 10:
        finished_flag = True
        break
        env.close()
        print('连续成功10轮')













