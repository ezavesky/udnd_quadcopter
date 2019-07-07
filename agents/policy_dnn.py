import numpy as np
from collections import namedtuple, deque

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agents.policy_search import PolicySearch_Agent

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
LR = 1e-2 # 5e-4               # learning rate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Policy, self).__init__()
        #self.seed = torch.manual_seed(seed)
        #self.fc1 = nn.Linear(state_size, fc1_units)
        #self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.fc3 = nn.Linear(fc2_units, action_size)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
        #x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        #return self.fc3(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        # print("probs --- ", action.item(), probs)
        return action.item(), m.log_prob(action)
    

class PolicyNN_Agent(PolicySearch_Agent):
    """Custom agent for NN policy learning with derivations from policy search skeleton. 
        Interacts with and learns from the environment (a task).  During development, two modes
        were created for different action set definitions.
        
        **ACTION_ONE_ROTOR==True**
        This DQN agent creates a set of integer actions based on the length of `self.action_scalars`
        and uses a duplicated network set to generate the right number of action (according to task)
        outputs.  This allows learning of a *single* model that shares some joint knowledge of 
        how to choose actions for different actors.
        
        To track the specific actor, states in the memory buffer are augmented by one dimension 
        to include the specific actor utilized.

        **ACTION_ONE_ROTOR==False**
        This DQN agent creates a set of integer actions based on the exhaustive combination of 
        different action values for use in a single policy model.  Specifically, it quantizes to derive
        the possible number of action values and replicates that for the total number of actors.
        
        To give each actor a full set of action values, an action space is created for each on/off 
        permutation.  For this reason, it's wise to keep *action_steps* rather low.
        """

    def __init__(self, task, seed=None, num_episodes=None, action_steps=4, reward_decay=1):
        """Initialize an Agent object.
        
        Params
        ======
            task (Task): task for environmnet definition
            seed (int): random seed
        """        

        super().__init__(task, seed, num_episodes)
        self.total_actions = pow(2, self.action_size)
        self.policy = Policy(self.state_size, self.total_actions, seed).to(device)
        # formulate policy engine with larger scalar set
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)


    #########################################################################
    # functions below this line should be adapted by derived classes
    #########################################################################

    def local_init(self):
        pass
        
    def local_reset_episode(self, include_randomize=False):
        self.error = 0
        self.memory = []
        self.rewards = []

    def local_peek_state(self):
        # for simplicity, just show final decision layer for action
        return (self.policy.fc2.weight)

    def local_step(self, reward, done, action, state, next_state):        
        # loop through each action required for individual memory insertion, applying discount
        self.rewards.append(reward)
        for i_actor in range(len(self.log_prob_action)):
            self.memory.append(-self.log_prob_action[i_actor] * self.gamma_last * reward)

    def local_learn(self, best_update):
        #print("--- loss ---")
        #print(self.policy_loss)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(self.memory).sum()
        policy_loss.backward()
        self.error = policy_loss.sum()
        # self.error = np.mean(self.rewards) 
        self.optimizer.step()
        
    def local_act(self, state):
        # Choose action based on given state and policy (to be overriden)
        # Returns actions for given state as per current policy.
        action, log_prob = self.policy.act(state)
        self.log_prob_action = [log_prob]
                
        # print("---- act ----")
        # print(action, self.gamma_last, log_prob, self.task.state()['position'])
        return action

    