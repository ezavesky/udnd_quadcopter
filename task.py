import numpy as np
from physics_sim import PhysicsSim
from scipy.spatial.distance import cdist as cdist_scipy

PENALTY_POSITION = 0.05
PENALTY_VELOCITY = 0.03
PENALTY_ORIENTATION = 0.005
REWARD_BASE = 1000
REWARD_VIOLATION = 0
ACTION_REPEAT = 1

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.runtime = runtime
        self.init_angle_velocities = init_angle_velocities
        self.sim = PhysicsSim(init_pose, init_velocities, self.init_angle_velocities, self.runtime) 
        self.action_repeat = ACTION_REPEAT # changed from previous setting of 3 to give more dynamic response
        self.target_pos = target_pos

        self.action_size = 4
        self.state_size = self.action_repeat * (len(self.sim.pose) + self.action_size)
        self.action_low = 0
        self.action_high = 900
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        # randomize start
        self.randomize()
        
    def randomize(self):
        """Stub for randomizing initial positions"""
        # Simulation
        # self.sim = PhysicsSim(self.init_pose, self.init_velocities, self.init_angle_velocities, self.runtime) 
        return self.reset()

    def _dist_func(self, v1, v2):
        """Single place for distance function compute!"""
        return np.sqrt(np.power(v1 - v2, 2).sum())  # squared eucliean (this breaks penalty compute in reward)
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = REWARD_BASE + self.get_reward_local() \
            - REWARD_BASE*PENALTY_POSITION * self._dist_func(self.sim.pose[:3], self.target_pos)  \
            - REWARD_BASE*PENALTY_ORIENTATION * self._dist_func(self.sim.pose[3:], [0] * 3 )
        return max(0, reward)
    
    def get_reward_local(self):
        return 0
            
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            # if done:
            #    for ii_position in range(3):  # special crash indicator
            #        if self.sim.lower_bounds[ii_position] == self.sim.pose[ii_position]:
            #            reward = 0
            pose_all.append(self.sim.pose)
            pose_all.append(rotor_speeds)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # modify to reset state to pose and zero rotor speed
        state = np.array((list(self.sim.pose) + [0]*self.action_size) * self.action_repeat) 
        return state
    
    def state(self):
        """Return a human-readable state for faster diagnosis"""
        return {"position":self.sim.pose[:3], "velocity":self.sim.v, 
                #"dist_init":self._dist_func(self.target_pos, self.sim.init_pose[:3]),
                #"dist_final":self._dist_func(self.target_pos, self.sim.pose[:3])
                "dist_init":abs(self.target_pos - self.sim.init_pose[:3]),
                "dist_final":abs(self.target_pos - self.sim.pose[:3])
               }
    
        
class HoverTask(Task):
    """HoverTask (environment) that randomly initializes position and needs a moderately steady hover at 2,2,2."""
    def __init__(self, init_pose=None, max_velocity=0.5, init_movement=0, target_pos=None, runtime=5.):
        """Initialize a HoverTask object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            max_velocity: max magnitude of final movement on hover (modulates reward)
            init_movement: max magnitude of initial movement (max of 5) for random generation
            target_pos: target/goal (x,y,z) position for the agent
        """
        self.init_pose = init_pose
        self.init_movement = init_movement
        
        self.max_velocity = max_velocity  # store max velocity
        if init_movement > 5: # rail/max the init random movement
            init_movement = 5
        
        if target_pos is None:
            target_pos = np.array([2., 2., 2.])                  # final pose
        # pass init to normal task for operation
        super().__init__(init_pose=init_pose, target_pos=target_pos, runtime=runtime) 

    def get_reward_local(self):
        """Add absolute error from max return reward. (equally important as position info)"""
        rail_velocity = abs(self.sim.v - self.max_velocity)
        rail_velocity[rail_velocity < 0] = 0
        return -REWARD_BASE*PENALTY_VELOCITY * np.sqrt(np.power(rail_velocity,2).sum())

    def randomize(self):
        """Randomizing initial position and velocity"""
        # randomize the pose and velocity
        init_pose = self.init_pose
        if init_pose is None:
            max_offset = 10
            init_pose = np.random.rand(1,6)[0]*max_offset-(max_offset/2.0)          # initial pos
            if init_pose[2] < 0 or (self.target_pos is not None and self.target_pos[2]>init_pose[2]):
                init_pose[2] += (max_offset/2.0)
            init_pose[3:] = 0   # zero out the euler angles
        init_velocities = np.random.rand(1,3)[0]*self.init_movement-(self.init_movement)/2    # initial velocities
        # Simulation re-init
        self.sim = PhysicsSim(init_pose, init_velocities, self.init_angle_velocities, self.runtime) 
        return super().randomize()


class LandingTask(HoverTask):
    """LandingTask (environment) stringent version of hover task with near-zero final velocity and landing in 0,0,0."""
    def __init__(self, init_pose=None, init_movement=0, runtime=5):
        """Initialize a LandingTask object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
        """
        target_pos = np.array([0., 0., 0.])                  # final pose
        # pass init to normal task for operation
        super().__init__(init_pose=init_pose, max_velocity=0.01, init_movement=init_movement, 
                         target_pos=target_pos, runtime=runtime)
