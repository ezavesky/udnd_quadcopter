import numpy as np
import random
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
ACTION_ONE_ROTOR = False # determines how actions are computed

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
        print(action.item(), probs)
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
        self.seed = random.seed(seed)
        # auto-manage gamma (discount)
        self.gamma_start = 1.0
        self.gamma_end = 0.01
        # constant decay proportional to episodes (about 0.995 over 1000 episodes)
        if reward_decay is None:
            self.gamma_decay = 1 - 0.005*1000/num_episodes
        else:
            self.gamma_decay = reward_decay
        self.gamma_last = self.gamma_start
        
        super().__init__(task)

        self.action_rows = task.action_repeat

        val_min = min(self.action_low, self.action_high)
        val_max = max(self.action_low, self.action_high)
        val_range = val_max - val_min
            
        # get number of discrete states (keep action size smaller)
        self.action_scalars = np.logspace(np.log(1), -np.log(val_range), 
                                      num=action_steps-1, base=2, endpoint=True) * val_range
        self.action_scalars = list(self.action_scalars) + [0]
        print("Action Gradients: {:}, Max: {:}".format(self.action_scalars, val_range))

        # policy network
        if ACTION_ONE_ROTOR:
            self.policy = Policy(self.state_size+1, len(self.action_scalars), seed).to(device)
        else:
            self.action_scalars_raw = self.action_scalars
            # in this mode, we compute exhaustive sclar array for action values
            action_size_total = pow(self.action_size, len(self.action_scalars))
            # this will be the look-up array for actual values
            self.action_scalars = np.zeros((action_size_total, self.action_size))
            # now fill in the action values for each of the actions
            value_mask = len(self.action_scalars_raw) - 1
            for ii_action in range(action_size_total):
                action_value = ii_action  # we'll keep shifting this one down
                for ii_actor in range(self.action_size):
                    lookup_idx = action_value & value_mask
                    #print(ii_action, ii_actor, action_value, value_mask, lookup_idx, self.action_scalars_raw[lookup_idx])
                    self.action_scalars[ii_action,ii_actor] = self.action_scalars_raw[lookup_idx]
                    action_value >>= len(self.action_scalars_raw)
            # formulate policy engine with larger scalar set
            self.policy = Policy(self.state_size, action_size_total, seed).to(device)
            print("Action Gradients: {:}, Max: {:}, Len: {:}".format(self.action_scalars, val_range, len(self.action_scalars)))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        
        # Replay memory
        self.memory = []
        self.policy_loss = None
        
        # save log prob each action
        self.log_prob_action = []

    def reset_episode(self, include_randomize=False):
        self.t_step = 0
        self.gamma_last = self.gamma_start
        self.memory = []
        self.error = 0
        return super().reset_episode(include_randomize)
    
    def peek_state(self):
        # for simplicity, just show final decision layer for action
        return (self.policy.fc2.weight)

    def _state_actor_stamp(self, state, actor_id):
        if ACTION_ONE_ROTOR:
            return np.concatenate((state, [actor_id]))
        return state
    
    def step(self, reward, done, action, state, next_state):
        super().step(reward, done, action, state, next_state)

        # curtail episolon
        self.gamma_last = max(self.gamma_end, self.gamma_decay*self.gamma_last) # decrease gamma
        
        # loop through each action required for individual memory insertion
        for i_actor in range(len(self.log_prob_action)):
            self.memory.append(-self.log_prob_action[i_actor] * self.gamma_last * reward)
        self.policy_loss = torch.cat(self.memory).sum()
        
    def learn_update(self, best_update):
        #print("--- loss ---")
        #print(self.policy_loss)
        self.optimizer.zero_grad()
        self.policy_loss.backward()
        self.error = self.policy_loss.sum()
        self.optimizer.step()
        self.memory = []
        
    def act(self, state, eps=None):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        # add specific actor to new state as we iterate over actors
        ret_act = [0] * self.action_size
        if ACTION_ONE_ROTOR:
            self.log_prob_action = [0] * self.action_size
        else:
            self.log_prob_action = [0]

        # loop through each action required for individual memory insertion
        for i_actor in range(len(self.log_prob_action)):
            # augment the state to have each actor's identity
            state_actor = self._state_actor_stamp(state, i_actor)
            # retreive action for this specific actor
            act_new, log_prob = self.actor_act(state_actor)
            # map to specific action
            if ACTION_ONE_ROTOR:
                ret_act[i_actor] = self.action_scalars[act_new]
            else:
                ret_act = self.action_scalars[act_new]
            self.log_prob_action[i_actor] = log_prob
                
        # print("---- act ----")
        # print(ret_act, self.gamma_last, self.error, self.task.state()['position'])
        return ret_act
    
    def actor_act(self, state, eps=None):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action, log_prob = self.policy.act(state)
        return action, log_prob

