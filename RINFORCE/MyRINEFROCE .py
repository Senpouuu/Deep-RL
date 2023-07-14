from collections import deque,namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'log_porb', 'reward'))
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

class sample_MDP():
    def __init__(self):
        self.MDP_buffer = deque()
    def MDP_buffer_append(self,state,action,log_prob,reward):
            self.MDP_buffer.append(Transition(state,action,log_prob,reward))
    def Cal_G(self,t,gamma = 0.99):
        discount = 1
        Gt = 0
        begin = t
        end = self.MDP_buffer.__len__()
        if begin > end:
            return
        for index in range(begin, end):
            Gt = Gt + (gamma ** discount) * self.MDP_buffer[index].reward
            discount = discount + 1
        return Gt

class policy(nn.Module):
    def __init__(self,n_states,n_actions):
        super(policy,self).__init__()
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
        state = torch.from_numpy(state).float().unsqueeze(0)
        # probs, (1, 2)
        #         probs = self.forward(autograd.Variable(state))
        probs = self.forward(state)
        # 以概率采样
        highest_prob_action = np.random.choice(self.n_actions, p=np.squeeze(probs.detach().numpy()))
        # prob => log prob

        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        # log_p < 0
        return highest_prob_action, log_prob

class value(nn.Module):
    def __init__(self,n_states):
        super(value,self).__init__()
        self.value_network = nn.Sequential(
            nn.Linear(n_states,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,1),
        )
    def forward(self,x):
        return self.value_network(x)


class Agent():
    def __init__(self, n_states, n_actions,eta=0.5, gamma=0.99):
        self.eta = eta
        self.gamma = gamma
        self.policy_model = policy(n_states,n_actions)
        self.value_model = value(n_states)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=0.001)
        self.MDP = sample_MDP()

    def train(self):
        batch = Transition(*zip(*self.MDP.MDP_buffer))
        Gt_batch = []
        for x in range(len(batch.reward)):
            Gt_batch.append(self.MDP.Cal_G(x,0.9))
        St_batch = []

        for x in range(len(batch.state)):
            St_batch.append(batch.state[x])

        Gt_batch = torch.tensor(Gt_batch)
        St_batch = torch.tensor(St_batch)


        # train value-network
        # value eval mode
        self.value_model.eval()

        # value pred
        V_pred = self.value_model(St_batch)


        # value train mode
        self.value_model.train()

        value_loss = F.mse_loss(Gt_batch,V_pred)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()



        # train policy-network
        # policy eval mode
        self.policy_model.eval()

        # baseline Function
        V_batch = self.value_model(St_batch).detach().squeeze()

        # Advantage
        A_batch = Gt_batch - V_batch
        # log_probs
        Lp_batch = []
        for x in range(len(batch.log_porb)):
            Lp_batch.append(batch.log_porb[x])


        # policy_grads
        Pg_batch = []
        for Lp, At in zip(Lp_batch, A_batch):
            Pg_batch.append(-Lp * At)


        # policy train mode
        self.policy_model.train()
        self.policy_optimizer.zero_grad()
        policy_grad = torch.stack(Pg_batch).sum()
        policy_grad.backward(retain_graph=True)
        self.policy_optimizer.step()




import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gym
env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = Agent(n_states,n_actions)
max_steps = 200
max_episodes = 5000


for episodes in range(max_episodes):
    state = env.reset()
    for steps in range(max_steps):
        action,log_prob = agent.policy_model.choose_action(state)
        next_state,reward,done,_ = env.step(action)
        agent.MDP.MDP_buffer_append(state,action,log_prob,reward)

        if done:
            agent.train()
            agent.MDP.MDP_buffer.clear()
            print(f'episode: {episodes}, length: {steps}')
            break
        state = next_state












