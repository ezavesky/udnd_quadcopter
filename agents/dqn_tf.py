import numpy as np
import random
from collections import namedtuple, deque

import tensorflow as tf

from agents.policy_search import PolicySearch_Agent

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, 
                 learning_rate=0.01, name='QNetwork'):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.float32, [None], name='actions')
            all_actions_ = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, fc1_units)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, fc2_units)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, all_actions_), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)    


class DQN_Agent(PolicySearch_Agent):
    """Custom agent with derivations from policy search skeleton. 
        Interacts with and learns from the environment (a task)."""

    def __init__(self, task, seed=None, num_episodes=None):
        """Initialize an Agent object.
        
        Params
        ======
            task (Task): task for environmnet definition
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        # constant decay proportional to episodes (about 0.995 over 1000 episodes)
        self.eps_decay = 1 - 0.005*1000/num_episodes
        
        super().__init__(task)
        
        tf.reset_default_graph()
        
        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed, 
                                       learning_rate=self.eps_decay, name="QNet_Local")
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed,
                                       learning_rate=self.eps_decay, name="QNet_Target")

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def reset_episode(self, include_randomize=False):
        self.t_step = 0
        return super().reset_episode(include_randomize)

    def learn(self):
        """Keep parent's learning of completing when done..."""
        return super().learn()
    
    def learn_dqn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states)   # pull all act values
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_expected = self.qnetwork_local(states)
        if False:
            print("-- Q states --")
            print(Q_targets)
            print(Q_expected)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)    
        
        
        
        # Now train with experiences
        saver = tf.train.Saver()
        rewards_list = []
        with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        step = 0
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                # Uncomment this next line to watch the training
                # env.render() 

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
                if explore_p > np.random.rand():
                    # Make a random action
                    action = env.action_space.sample()
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.reset()
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = env.step(env.action_space.sample())

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network
                target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = (0, 0)

                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                               mainQN.targetQs_: targets,
                                               mainQN.actions_: actions})
        
    saver.save(sess, "checkpoints/cartpole.ckpt")

    
    
    
    
    
    
    def step(self, reward, done, action, state, next_state):
        super().step(reward, done, action, state, next_state)
    
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # curtail episolon
        self.eps_last = max(self.eps_end, self.eps_decay*self.eps_last) # decrease epsilon
        
        # Learn every UPDATE_EVERY time step.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn_dqn(experiences, GAMMA)
                
    def act(self, state, eps=None):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #state = torch.from_numpy(state).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # use self-managed epsilon
        if eps is None: 
            eps = self.eps_last            
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            # return full action space, not just a max
            #ret_act = np.argmax(action_values.cpu().data.numpy())
            ret_act = action_values.cpu().data.numpy()
        else:
            # return random for each action size, not just numeric state
            ret_act = np.random.rand(1, self.action_size)         
            # return random.choice(np.arange(self.action_size))
        return ret_act[0]


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
        idx = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
