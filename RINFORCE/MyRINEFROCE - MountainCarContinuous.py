'Due to my technical reasons, this code cannot converge.If you have any good ideas,welcome to contact me or give some issues.'
from collections import deque,namedtuple
Transition = namedtuple('Transition', ('state','log_porb','reward'))
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
import torch.distributions as dist


class sample_MDP():
    def __init__(self):
        self.MDP_buffer = deque()
    def MDP_buffer_append(self,state,log_prob,reward):
            self.MDP_buffer.append(Transition(state,log_prob,reward))
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

class gass_policy(nn.Module):
    def __init__(self,n_states,output_size = 2):
        super(gass_policy,self).__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(n_states,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,output_size),
            nn.Softmax()
        )
    def forward(self,x):
        return self.policy_network(x).unsqueeze(0)


    def gaussian(self, mu, sigma):

        normal_dist = dist.Normal(mu, sigma)
        a = normal_dist.rsample()
        prob_density = normal_dist.log_prob(a)
        return a.detach().numpy(),prob_density

class value(nn.Module):
    def __init__(self,n_states):
        super(value,self).__init__()
        self.value_network = nn.Sequential(
            nn.Linear(n_states,32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,1),
        )
    def forward(self,x):
        return self.value_network(x)


class Agent():
    def __init__(self, n_states,eta=0.5, gamma=0.99):
        self.eta = eta
        self.gamma = gamma
        self.policy_model = gass_policy(n_states,2)
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
        policy_grad.backward()
        self.policy_optimizer.step()




import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gym
env = gym.make('MountainCarContinuous-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
agent = Agent(n_states,n_actions)
max_steps = 200
max_episodes = 5000


for episodes in range(max_episodes):
    state = env.reset()
    for steps in range(max_steps):
        mu_log_sigma = agent.policy_model(torch.tensor(state))
        mu = mu_log_sigma[:,0]
        #log_sigma = mu_log_sigma[:,1]
        #sigma = torch.sqrt(torch.exp(log_sigma))
        sigma = mu_log_sigma[:, 1]
        action,log_prob = agent.policy_model.gaussian(mu,sigma)
        next_state,reward,done,_ = env.step(action)
        agent.MDP.MDP_buffer_append(state,log_prob,reward)

        if steps > 198:
            agent.train()
            agent.MDP.MDP_buffer.clear()
            print(f'episode: {episodes}, steps: {steps},state: {state} , action: {action}  , done:{done}')

        if done:
            print("GOOD!")
            break

        state = next_state












