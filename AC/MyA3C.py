from collections import deque,namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward' , 'log_porb'))
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

class sample_MDP:
    def __init__(self):
        self.MDP_buffer = deque()

    def MDP_buffer_append(self,state,action,next_state,reward,log_prob):
            self.MDP_buffer.append(Transition(state,action,next_state,reward,log_prob))

class Actor(nn.Module):
    def __init__(self,n_states,n_actions):
        super(Actor,self).__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(n_states,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,n_actions),
            nn.Softmax()
        )
        self.n_actions = n_actions
        self.n_states = n_states
    def forward(self,x):
        return self.policy_network(x)

    def choose_action(self, state):
        # given state
        # state.shape (4, ), 1d numpy ndarray
        # state, (1, 4)
        # probs, (1, 2)
        # probs = self.forward(autograd.Variable(state))
        probs = self.forward(state)

        # 以概率采样
        highest_prob_action = np.random.choice(self.n_actions, p=np.squeeze(probs.detach().numpy()))
        # prob => log prob

        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        # log_p < 0
        return highest_prob_action, log_prob

class Critic(nn.Module):
    def __init__(self,n_states):
        super(Critic,self).__init__()
        self.value_network = nn.Sequential(
            nn.Linear(n_states,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,1),
        )
    def forward(self,x):
        return self.value_network(x)


class Global_Agent():
    def __init__(self, n_states, n_actions):
        self.actor_model = Actor(n_states,n_actions)
        self.critic_model = Critic(n_states)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=1e-3)

    def run(self, max_steps=200, max_episodes=200):
        env = gym.make('CartPole-v0')
        for episodes in range(max_episodes):
            state = env.reset()
            state = torch.from_numpy(state).unsqueeze(0)
            for steps in range(max_steps):
                # env.render()
                action, log_prob = self.actor_model.choose_action(state)
                next_state, _, done, _ = env.step(action)
                if done:
                    next_state = None
                else:
                    next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                    next_state = next_state.unsqueeze(0)
                state = next_state

                if done:
                    print(f'episode: {episodes}, length: {steps}, done:{done}')
                    break


class Agent():
    def __init__(self, n_states, n_actions,eta=0.5, gamma=0.9, batch_size = 32):
        self.eta = eta
        self.gamma = gamma
        self.actor_model = Actor(n_states,n_actions)
        self.critic_model = Critic(n_states)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=1e-3)
        self.batch_size = batch_size
        self.MDP = sample_MDP()

    def choose(self,mask,taget,choose,choose_index = 0):
        for index, x in enumerate(mask):
            if x == 1:
                taget[index] = choose.squeeze()[choose_index]
                choose_index = choose_index + 1
            else:
                taget[index] = torch.tensor(0)
        return taget


    def train(self,global_agent):

        if self.MDP.MDP_buffer.__len__() < self.batch_size:
            return

        batch = Transition(*zip(*self.MDP.MDP_buffer))


        # 32 * 4
        state_batch = torch.cat(batch.state)

        # 32
        reward_batch = torch.cat(batch.reward)

        # <32 * 4
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])


        # train critic_model
        # critic eval mode
        self.critic_model.eval()

        # V_pred
        # 32 * 1
        critic_state_values = self.critic_model(state_batch)

        # yt
        non_none_musk = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        next_state_values = torch.zeros(self.batch_size)

        # V_next
        # 32 * 1
        next_state_values = self.choose(non_none_musk, next_state_values,self.critic_model(next_state_batch).squeeze().detach())

        critic_yt_batch = reward_batch + self.gamma * next_state_values

        # critic train mode
        self.critic_model.train()

        loss = F.mse_loss(critic_yt_batch, critic_state_values)
        self.critic_optimizer.zero_grad()
        loss.backward()
        for local_para,global_para in zip (self.critic_model.parameters(),global_agent.critic_model.parameters()):
            global_para._grad = local_para._grad
        global_agent.critic_optimizer.step()

        self.critic_model.load_state_dict(global_agent.critic_model.state_dict())

        # actor eval mode
        self.actor_model.eval()

        # V_pred
        actor_state_values = self.critic_model(state_batch).detach()

        # yt
        actor_yt_batch = reward_batch + self.gamma * next_state_values

        # Advantage
        At_batch = actor_yt_batch - actor_state_values

        # log_probs
        Lp_batch = []
        for x in range(len(batch.log_porb)):
            Lp_batch.append(batch.log_porb[x])

        # policy_grads
        Pg_batch = []
        for Lp, At in zip(Lp_batch, At_batch):
            Pg_batch.append(-Lp * At)

        # actor train mode
        self.actor_model.train()
        self.actor_optimizer.zero_grad()
        policy_grad = torch.stack(Pg_batch).sum()
        policy_grad.backward()
        for local_para,global_para in zip(self.actor_model.parameters(),global_agent.actor_model.parameters()):
            global_para._grad = local_para._grad

        global_agent.actor_optimizer.step()
        self.actor_model.load_state_dict(global_agent.actor_model.state_dict())

        self.MDP.MDP_buffer.clear()

    def act(self,global_agent,max_steps = 200,max_episodes = 500):
        info = []
        env = gym.make('CartPole-v0')
        for episodes in range(max_episodes):
            state = env.reset()
            state = torch.from_numpy(state).unsqueeze(0)
            for steps in range(max_steps):
                # env.render()
                action, log_prob = self.actor_model.choose_action(state)
                next_state, _, done, _ = env.step(action)

                if done:
                    if steps < 195:
                        reward = torch.FloatTensor([-1.])

                    else:
                        reward = torch.FloatTensor([1.])
                    next_state = None


                else:
                    reward = torch.FloatTensor([0])
                    next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                    next_state = next_state.unsqueeze(0)

                self.MDP.MDP_buffer_append(state, action, next_state, reward, log_prob)
                self.train(global_agent)
                state = next_state

                if done:
                    info.append(f'episode: {episodes}, length: {steps}, done:{done}')
                    break
        return info

import gym
env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
global_agent = Global_Agent(n_states,n_actions)

# 多线程执行若干local_agentx...

local_agent1 = Agent(n_states,n_actions)
local_agent1.act(global_agent)
global_agent.run()



