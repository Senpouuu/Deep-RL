'Due to my technical reasons, this code cannot converge.If you have any good ideas,welcome to contact me or give some issues.'

from collections import deque,namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
import random
import copy


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

class Critic(nn.Module):
    def __init__(self,n_states,n_actions):
        super(Critic, self).__init__()
        self.Critic_Network = nn.Sequential(
            nn.Linear(n_states + n_actions, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self,states,actions):
        return self.Critic_Network(torch.cat((states, actions), 1))



class Actor(nn.Module):
    def __init__(self,n_states, n_actions,max_actions = 1):
        super(Actor,self).__init__()
        self.max_actions = max_actions
        self.Actor_Network = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return torch.tanh(self.Actor_Network(x)) * self.max_actions

class Agent:
    def __init__(self, n_states, n_actions,freedom_size,tau=0.005, gamma=0.99, replay_buffer_size=10000, batch_size=10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.tau = tau
        self.gamma = gamma
        self.replay_buffer = experience_replay(replay_buffer_size,batch_size)

        self.Actor_model = Actor(n_states,freedom_size)
        self.Critic_model = Critic(n_states,n_actions)
        self.Actor_Optimizer = optim.Adam(self.Actor_model.parameters(),lr = 1e-4)
        self.Critic_Optimizer = optim.Adam(self.Critic_model.parameters(),lr = 1e-4)

        self.Actor_target_model = copy.deepcopy(self.Actor_model)
        self.Critic_target_model = copy.deepcopy(self.Critic_model)

        self.batch_size = batch_size


    def train(self):

        if self.batch_size > self.replay_buffer.len():
            return

        batch = self.replay_buffer.batch_sample()

        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state)

        action_batch = torch.cat(batch.action)

        next_state_batch = torch.cat(batch.next_state)

        reward_batch = torch.cat(batch.reward)


        # train critic network
        # eval mode
        self.Critic_model.eval()

        # Q_pred(s,a)
        state_action_values = self.Critic_model.forward(state_batch,action_batch)

        # next_Q_pred
        next_state_action_values = self.Critic_target_model.forward(next_state_batch,self.Actor_target_model.forward(next_state_batch)).squeeze().detach()

        target_state_action_values = reward_batch + (self.gamma * next_state_action_values)

        # train mode
        self.Critic_target_model.train()

        critic_loss = F.mse_loss(state_action_values,target_state_action_values.unsqueeze(1))
        self.Critic_Optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_Optimizer.step()


        # train actor network
        # train mode
        self.Actor_model.train()
        actor_loss = -self.Critic_model.forward(state_batch,self.Actor_model.forward(state_batch)).mean()       # 评估在当前状态下Actor网络给出下一个动作的好坏
        self.Actor_Optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_Optimizer.step()


    def softUpdate(self):
        for param,param_t in zip(self.Critic_model.parameters(),self.Critic_target_model.parameters()):
            param_t.data.copy_(self.tau * param.data + (1 - self.tau) * param_t.data)

        for param,param_t in zip(self.Actor_model.parameters(),self.Actor_target_model.parameters()):
            param_t.data.copy_(self.tau * param.data + (1 - self.tau) * param_t.data)



import gym
env = gym.make("MountainCarContinuous-v0")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
agent = Agent(n_states,n_actions,n_actions)
max_episodes = 200
max_steps = 199


for episodes in range(max_episodes):
    state = env.reset()
    state = torch.from_numpy(state).unsqueeze(0)
    total_rewards = 0

    for steps in range(max_steps):
        #env.render()
        action = agent.Actor_model(state).detach()
        action = action.numpy()
        next_state,reward,done,_ = env.step(action)
        total_rewards += reward
        if done:
            print('SUCCESS！')
            break

        else:
            next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
            next_state = next_state.squeeze().unsqueeze(0)

        reward = torch.tensor([reward])
        action = torch.from_numpy(action)
        agent.replay_buffer.replay_buffer_append(state,action,next_state,reward)
        state = next_state
        agent.train()
        agent.softUpdate()

    print(f'episode: {episodes}, steps: {steps},state: {state} , action: {action} , rewards:{total_rewards} , done:{done}')


