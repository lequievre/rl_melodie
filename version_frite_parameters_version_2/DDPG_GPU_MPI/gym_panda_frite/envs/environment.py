import pybullet as p
import pybullet_data
import threading
import time
import os, inspect

class Environment():
	
	def __init__(self, json_decoder=None, gui = True):
		
		if json_decoder==None:
			raise RuntimeError("=> Environment class need a JSON Decoder, to get some parameters !!!")
			return
			
		self.time_step = json_decoder.config_data["env"]["time_step"]
		self.n_substeps = json_decoder.config_data["env"]["n_substeps"]
		self.gui = gui
		
		self.time_step_sleep = json_decoder.config_data["env"]["time_step_sleep"]
		self.running = False
		
		p.connect(p.GUI if self.gui else p.DIRECT)
		#p.setPhysicsEngineParameter(enableFileCaching=0)
		
		self.step_thread = threading.Thread(target=self.step_simulation)
		self.step_thread.daemon = True
		self.step_thread.start()
		
		
	def step_simulation(self):
		
		p.setTimeStep(self.time_step)
		while True:
			if self.running:
				p.stepSimulation()
			time.sleep(self.time_step_sleep)
	
	def pause(self):
		self.running = False
		
		
	def start(self):
		self.running = True
			
	def reset(self):
		
		self.pause()
		
		# reset pybullet to deformable object
		p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

		# bullet setup
		# add pybullet path
		currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		#print("currentdir = {}".format(currentdir))
		p.setAdditionalSearchPath(currentdir)

		#p.setPhysicsEngineParameter(numSolverIterations=150, numSubSteps = self.n_substeps)
		p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)

		# Set Gravity to the environment
		p.setGravity(0, 0, -9.81)
		#p.setGravity(0, 0, 0)
		
		p.setTimeStep(self.time_step)
		
		self.start()
	
		
