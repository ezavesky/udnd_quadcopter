import numpy as np
from task import Task
import random

class PolicySearch_Agent():
    def __init__(self, task, seed=None, num_episodes=None):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.action_delta = self.action_range * 0.5

        # auto-manage gamma (discount)
        self.gamma_start = 1.0
        self.gamma_end = 0.01
        # constant decay proportional to episodes (about 0.995 over 1000 episodes)
        if num_episodes is not None:
            self.gamma_decay = 1 - 0.005*1000/num_episodes
        else:
            self.gamma_decay = 1
        self.gamma_last = self.gamma_start
        self.seed = random.seed(seed)

        self.local_init()
        
        # Episode variables (reset with random once to reset `best`)
        self.reset_episode(True)

    def reset_episode(self, include_randomize=False):
        self.total_reward = 0.0  # total reward
        self.count = 0  # count for number of steps
        self.error = 0  # secondary error (similar to reward)
        self.action_values = np.array([self.action_low] * self.action_size)  # the actual action values 
        self.gamma_last = self.gamma_start

        self.local_reset_episode(include_randomize)
        
        # Score tracker and learning parameters
        if include_randomize:
            self.best_w = None
            self.best_score = -np.inf
            self.best_state = self.task.state()
            return self.task.randomize()
        return self.task.reset()
      
 
    def step(self, reward, done, action, state, next_state):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1
        # curtail episolon
        self.gamma_last = max(self.gamma_end, self.gamma_decay*self.gamma_last) # decrease gamma

        # update overall action values based on action choice
        action_deref = self.act_value_map(action, False)
        self.action_values = action

        # call specific worker function
        self.local_step(reward, done, action_deref, state, next_state)
        
        # Learn, if at end of episode
        if done:
            self.learn()

    def act_value_map(self, act_input, forward_map=True):
        """Apply action values forward (e.g. scale up) or backward (e.g. scale back to simple action)"""
        if forward_map:  # add to speed, but also naturally spin down
            act_gate = np.unpackbits(np.array([[act_input]], dtype=np.uint8)).astype(float)[-self.action_size:]
            # print("--pre --", act_input, act_gate, self.action_values)
            nd_new_values = self.action_values + act_gate*self.action_delta - self.action_delta*0.5
            nd_new_values[nd_new_values < self.action_low] = self.action_low
            nd_new_values[nd_new_values > self.action_high] = self.action_high
            # print(nd_new_values)
        else:   # subtract to see what happened
            #print("--pre-- ", act_input, self.action_values)
            nd_new_values = act_input - self.action_values + self.action_delta*0.5
            nd_new_values /= self.action_delta
            nd_new_values = (8-self.action_size)*[0] + list(nd_new_values.astype(int))
            nd_new_values = np.packbits(nd_new_values)
            #print(nd_new_values, act_gate)
        return nd_new_values            
            
    def act(self, state):
        """Compute local action, adjust action values by application policy (in this class)"""
        int_act = self.local_act(state)
        return self.act_value_map(int_act), int_act       

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_state = self.task.state()
            self.local_learn(True)
        else:
            self.local_learn(False)
            
    #########################################################################
    # functions below this line should be adapted by derived classes
    #########################################################################

    def local_init(self):
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=1 ) # start producing actions in a decent range
        
    def local_reset_episode(self, include_randomize=False):
        if include_randomize:
            self.best_w = None
        self.noise_scale = 0.1
    
    def local_step(self, reward, done, action, state, next_state):
        pass
    
    def local_act(self, state):
        # Choose action based on given state and policy (to be overriden)
        act_values = np.dot(state, self.w)  # simple linear policy
        # print("predict -- ", act_values)
        act_values[act_values < 0] = 0
        act_values[act_values > 0] = 1
        
        action = (8-self.action_size)*[0] + list(act_values.astype(int))
        action = np.packbits(action)
        return action

    def local_learn(self, best_update):
        self.error = self.score
        if best_update or self.best_w is None:
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
            self.best_w = self.w
        else:
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
            self.w = self.best_w
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions

    def local_peek_state(self):
        return (self.w, self.noise_scale)
        