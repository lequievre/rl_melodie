from gym import Wrapper
import numpy as np

class WrapperPandaFriteEnvROS(Wrapper):
	
	def __init__(self, env):
		super().__init__(env)
		self.env = env
	
	def seed(self, seed=None):
		self.env.seed(seed)
			
	def go_to_home_position(self):
		self.env.go_to_home_position()
		
	def reset_ros(self, **kwargs):
		return self.env.reset_ros(**kwargs)

	def step_ros(self, action):
		observation, reward, done, info = self.env.step_ros(action)
		
		done = True
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		
		max_d = 0
		
		for i in range(nb_mesh_to_follow):
			#print("i= {}, observation before ={}\n".format(i,observation[(6+(i*3)):(6+(i*3)+3)]))
			#print("random observation = ", self.env.sample_random_observation())
			#observation[(6+(i*3)):(6+(i*3)+3)] += self.env.sample_random_observation()
			#print("i= {}, observation after ={}\n".format(i,observation[(6+(i*3)):(6+(i*3)+3)]))
			current_pos_mesh = observation[(6+(i*3)):(6+(i*3)+3)]
			goal_pos_id_frite = self.env.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			if (d > max_d):
				max_d = d
				
		info = {
			'is_success': self.env.is_success(max_d),
			'max_distance_error' : max_d,
		}

		reward = -max_d
		if (max_d > self.distance_threshold):
			done = False
		

		return observation, reward, done, info
		
