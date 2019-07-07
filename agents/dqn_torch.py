import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.policy_search import PolicySearch_Agent

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.9            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 5e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

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
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_Agent(PolicySearch_Agent):
    """Custom agent for DQN learning with derivations from policy search skeleton. 
        Interacts with and learns from the environment (a task).
        
        This DQN agent creates a set of integer actions based on the length of `self.action_scalars`
        and uses a duplicated network set to generate the right number of action (according to task)
        outputs.  This allows learning of a *single* model that shares some joint knowledge of 
        how to choose actions for different actors.
        
        To track the specific actor, states in the memory buffer are augmented by one dimension 
        to include the specific actor utilized.
        """

    def __init__(self, task, seed=None, num_episodes=None, eps=0.2):
        """Initialize an Agent object.
        
        Params
        ======
            task (Task): task for environmnet definition
            seed (int): random seed
        """
        super().__init__(task, seed, num_episodes)
        self.eps = eps

        self.total_actions = pow(2, self.action_size)
        
        # Q-Network        
        self.qnetwork_local = QNetwork(self.state_size, self.total_actions, seed).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.total_actions, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(self.total_actions, BUFFER_SIZE, BATCH_SIZE, seed)

    def actor_learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        if False:
            print("-- Q states --")
            print(Q_targets)
            print(Q_expected)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.error += loss
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.actor_soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def actor_soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        
        
    #########################################################################
    # functions below this line should be adapted by derived classes
    #########################################################################

    def local_init(self):
        pass
        
    def local_peek_state(self):
        # for simplicity, just show final decision layer for action
        return (self.qnetwork_target.fc3.weight)

    def local_learn(self, best_update):
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.actor_learn(experiences, GAMMA)
    
    def local_act(self, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #state = torch.from_numpy(state).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_t)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            # return full action space, not just a max
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            # return random for each action size, not just numeric state
            action = random.choice(np.arange(self.action_size))
        return action

    def local_reset_episode(self, include_randomize=False):
        pass
    
    def local_step(self, reward, done, action, state, next_state):
        # Save experience (for each actor) in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time step.
        if (self.count % UPDATE_EVERY) == 0:
            self.error = 0
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.actor_learn(experiences, GAMMA)
            
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)