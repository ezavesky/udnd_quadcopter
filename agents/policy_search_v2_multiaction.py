import numpy as np
from task import Task

class PolicySearch_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range
        
        # Episode variables (reset with random once to reset `best`)
        self.reset_episode(True)

    def reset_episode(self, include_randomize=False):
        self.total_reward = 0.0
        self.count = 0
        self.error = 0

        # Score tracker and learning parameters
        if include_randomize:
            self.best_w = None
            self.best_score = -np.inf
            self.best_state = self.task.state()
            self.noise_scale = 0.1
            return self.task.randomize()
        return self.task.reset()
    
    def step(self, reward, done, action, state, next_state):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_state = self.task.state()
            self.learn_update(True)
        else:
            self.learn_update(False)
            
    def learn_update(self, best_update):
        self.error = self.score
        if best_update or self.best_w is None:
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
            self.best_w = self.w
        else:
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
            self.w = self.best_w
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions

    def peek_state(self):
        return (self.w, self.noise_scale)
        