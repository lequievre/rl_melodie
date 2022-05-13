from gym import Wrapper
import numpy as np

class WrapperPandaFriteEnv(Wrapper):
	
	def __init__(self, env):
		super().__init__(env)
		self.env = env
	
	def seed(self, seed=None):
		self.env.seed(seed)
			
	def reset_env(self, use_frite=True):
		self.env.reset_env(use_frite)
		
	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		
		done = True
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		
		max_d = 0
		
		for i in range(nb_mesh_to_follow):
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
		
	def draw_env_box(self):
		self.env.draw_env_box()

	def action(self, action):
		#print(self.env.sample_random_action())
		action_return = action + self.env.sample_random_action()
		#print("TransformReward-> action={}, action_return={}".format(action,action_return))

		return action_return
