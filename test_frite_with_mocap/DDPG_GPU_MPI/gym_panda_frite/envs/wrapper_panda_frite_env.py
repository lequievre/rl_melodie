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
		observation, reward, done, info = self.env.step(self.action(action))

		return observation, reward, done, info
		
	def draw_env_box(self):
		self.env.draw_env_box()

	def action(self, action):
		#print(self.env.sample_random_action())
		action_return = action + self.env.sample_random_action()
		#print("TransformReward-> action={}, action_return={}".format(action,action_return))

		return action_return
