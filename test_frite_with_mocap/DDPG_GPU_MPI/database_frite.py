import sys
import pybullet as p
import numpy as np
from numpy import random
import math
import time

class Database_Frite:
	def __init__(self, json_decoder=None):
		
		if json_decoder==None:
			raise RuntimeError("=> Database_Frite class need a JSON Decoder, to get some parameters !!!")
			return
			
		self.load_name = json_decoder.config_data["database"]["name"]
		self.generate_name = 'generate_' + json_decoder.config_data["database"]["name"]
		self.path_load = json_decoder.config_dir_name
		self.path_generate = json_decoder.config_dir_name
		self.db_nb_random_goal = json_decoder.config_data["database"]["generate"]["nb_random_goal"]
		
		# type of db , 0 = classic, 1 = random, 2 = mocap
		self.type_db_generate = json_decoder.config_data["database"]["generate"]["type_db"]
		self.type_db_load = json_decoder.config_data["database"]["load"]["type_db"]
		
		self.nb_x = json_decoder.config_data["database"]["generate"]["nb_x"]
		self.nb_y = json_decoder.config_data["database"]["generate"]["nb_y"]
		self.nb_z = json_decoder.config_data["database"]["generate"]["nb_z"]
		
		self.time_action = json_decoder.config_data["env"]["time_set_action"]
		
		self.nb_lines = 0
		self.nb_deformations = 0
		
		self.env = None
		self.data = None
		self.nb_points = None
		
		print("Database_Frite -> nb_x={}, nb_y={}, nb_z={}".format(self.nb_x,self.nb_y,self.nb_z))
		print("Database_Frite -> type_db_load = {}, type_db_generate = {}".format(self.type_db_load, self.type_db_generate))
	
	def set_env(self, env):
		self.env = env
		self.nb_points = len(self.env.id_frite_to_follow)
	
	
	def debug_point(self, pt, offset = 0.1, width = 3.0, color = [1, 0, 0]):
		
		p.addUserDebugLine(lineFromXYZ          = [pt[0]+offset, pt[1], pt[2]]  ,
						   lineToXYZ            = [pt[0]-offset, pt[1], pt[2]],
						   lineColorRGB         = color  ,
						   lineWidth            = width        ,
						   lifeTime             = 0          )
						   
		p.addUserDebugLine(lineFromXYZ          = [pt[0], pt[1]+offset, pt[2]]  ,
						   lineToXYZ            = [pt[0], pt[1]-offset, pt[2]],
						   lineColorRGB         = color  ,
						   lineWidth            = width        ,
						   lifeTime             = 0          )
						   
	
	
	def debug_all_points(self):
		for i in range(self.nb_deformations):
			for j in range(self.nb_points):
				self.debug_point(pt=self.data[i,j], offset=0.01)
			
	
	
	def debug_all_random_points(self, nb):
		for i in range(nb):
			a_pt = self.get_random_targets()
			for j in range(self.nb_points):
				self.debug_point(pt = a_pt[j], offset =0.01, color = [0, 0, 1])
	
	def print_config_generate(self):
		self.init_spaces()
		
		if self.type_db_generate==1:
			print("RANDOM DB !")
			print("db_nb_random_goal = {}".format(self.db_nb_random_goal))
		elif self.type_db_generate==0:
			print("CLASSIC DB !")
			d_x = self.goal_high[0] - self.goal_low[0]
			d_y = self.goal_high[1] - self.goal_low[1]
			d_z = self.goal_high[2] - self.goal_low[2]
			
			print("****** CONFIG DATABASE ***************")
			print("nb_x={}, nb_y={}, nb_z={}".format(self.nb_x,self.nb_y,self.nb_z))
			print("d_x={}, d_y={}, d_z={}".format(d_x,d_y,d_z))
			print("step_x={}, step_y={}, step_z={}".format(self.step_x,self.step_y,self.step_z))
			print("range_x={}, range_y={}, range_z={}".format(self.range_x,self.range_y,self.range_z))
			print("delta_x={}, delta_y={}".format(self.delta_x,self.delta_y))
			print("**************************************")
		
	
	def init_spaces(self):
		
		self.gripper_position = self.env.get_gripper_position()
		self.goal_low = self.env.goal_space.low
		self.goal_high = self.env.goal_space.high
		
		#self.delta_x = math.ceil(((self.gripper_position[0]-self.goal_low[0])/(self.goal_high[0]-self.goal_low[0]))*self.nb_x) + 1
		#self.delta_y = math.ceil(((self.gripper_position[1]-self.goal_low[1])/(self.goal_high[1]-self.goal_low[1]))*self.nb_y) + 1
		
		
		self.step_x = float((self.goal_high[0]-self.goal_low[0])/(self.nb_x +1))
		self.step_y = float((self.goal_high[1]-self.goal_low[1])/(self.nb_y + 1))
		self.step_z = float((self.goal_high[2]-self.goal_low[2])/(self.nb_z + 1))
		
		print("step_x={}, step_y={}, step_z={}".format(self.step_x,self.step_y,self.step_z))
		
		
		self.delta_x = math.ceil((self.gripper_position[0]-self.goal_low[0])/self.step_x)
		self.delta_y = math.ceil((self.gripper_position[1]-self.goal_low[1])/self.step_y)
		
		print("delta_x={}, delta_y={}".format(self.delta_x,self.delta_y))

		self.range_x = self.nb_x + 1
		self.range_y = self.nb_y + 1
		self.range_z = self.nb_z + 1
		
		print("step_x={}, step_y={}, step_z={}".format(self.step_x,self.step_y,self.step_z))
			
		
	def get_random_targets(self):
		if self.data is not None:
			index = random.randint(self.nb_deformations-1)
			
			return self.data[index]
		else:
			return None
	
	def load(self):
		if (self.type_db_load==0 or self.type_db_load==1):
			self.load_classic_random()
		elif self.type_db_load==2:
			self.load_mocap()
		else:
			raise RuntimeError("=> Can load classic (type_db=0) or random db (type_db=1) or mocap db (type_db=2) !!!")
	
	
	def load_mocap(self):
		print("=> LOAD MOCAP DATABASE Name = {}".format(self.path_load + self.load_name))
		f = open(self.path_load + self.load_name)
		poses_mocap_array = np.zeros((5,3))
		total_list = []
		self.nb_lines = 0
		for line in f.readlines():
			line_split = line.split()
			self.nb_lines+=1
			
			if (len(line_split) != 22):
				print("Erreur on line ", nbline)
				
			else:
				#goal_x = float(line_split[0])
				#goal_y = float(line_split[1])
				#goal_z = float(line_split[2])
				# Get orientation x,y,z,w
				orientation_base_frame_array = np.array([float(line_split[3]),float(line_split[4]),float(line_split[5]),float(line_split[6])])
				
				for i in range(5):
					x = float(line_split[((i+1)*3)+4])
					y = float(line_split[((i+1)*3)+5])
					z = float(line_split[((i+1)*3)+6])
					poses_mocap_array[i][0] = x
					poses_mocap_array[i][1] = y
					poses_mocap_array[i][2] = z
				
				#print("poses_mocap_array = {}, orientation_base_frame_array = {}".format(poses_mocap_array,orientation_base_frame_array ))	
				poses_mocap_array_in_arm_frame = self.env.transform_mocap_poses_to_arm_poses(poses_mocap_array, orientation_base_frame_array)
				
				for i in reversed(range(4)):
					total_list.append(poses_mocap_array_in_arm_frame[i+1][0])
					total_list.append(poses_mocap_array_in_arm_frame[i+1][1])
					total_list.append(poses_mocap_array_in_arm_frame[i+1][2])
				
		self.nb_deformations = self.nb_lines
		print("nb lines = {}, nb points = {}, nb_deformations = {}".format(self.nb_lines, self.nb_points,self.nb_deformations))
		self.data = np.array(total_list).reshape(self.nb_deformations, self.nb_points, 3)
		
		
		print("shape = {}".format(self.data.shape))	
		print(self.data[0,0])
		
	def load_classic_random(self):
		print("=> LOAD DATABASE Name = {}".format(self.path_load + self.load_name))
		f = open(self.path_load + self.load_name)
		total_list = []
		for line in f.readlines():
			self.nb_lines+=1
			line_split = line.split()
			# 0 = X mean, 1 = Y mean, 2 = Z mean
			total_list.append(float(line_split[3])) #x shifted
			total_list.append(float(line_split[4])) #y shifted
			total_list.append(float(line_split[5]))	#z shifted

		self.nb_deformations = int(self.nb_lines/self.nb_points)
		
		print("nb lines = {}, nb points = {}, nb_deformations = {}".format(self.nb_lines, self.nb_points,self.nb_deformations))
		self.data = np.array(total_list).reshape(self.nb_deformations, self.nb_points, 3)
		
		
		print("shape = {}".format(self.data.shape))	
		print(self.data[0,0])
		
	
	def write_floats(self, f):
		self.env.compute_mesh_pos_to_follow(draw_normal=False)
		for k in range(len(self.env.position_mesh_to_follow)):
			f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(self.env.mean_position_to_follow[k][0], self.env.mean_position_to_follow[k][1], self.env.mean_position_to_follow[k][2], self.env.position_mesh_to_follow[k][0],  self.env.position_mesh_to_follow[k][1], self.env.position_mesh_to_follow[k][2]))

	def go_to_corner(self):
		# go to corner
		d_x_y_z= [-self.step_x, 0.0, 0.0]
		for x in range(self.delta_x):
			#print("x-> {}".format(d_x_y_z))
			self.env.set_action_cartesian(d_x_y_z)
			time.sleep(self.time_action)

		d_x_y_z= [0.0, -self.step_y, 0.0]
		for y in range(self.delta_y):
			#print("y-> {}".format(d_x_y_z))
			self.env.set_action_cartesian(d_x_y_z)
			time.sleep(self.time_action)


	def generate(self):
		if self.type_db_generate==0:
			self.generate_classic()
		elif self.type_db_generate==1:
			self.generate_random()
		else:
			raise RuntimeError("=> Can generate classic (type_db=0) or random db (type_db=1) !!!")
			
	def generate_random(self):
		print("->Open database file : ", self.path_generate + self.generate_name , " !")
		f = open(self.path_generate + self.generate_name, "w+")
		
		for i in range(self.db_nb_random_goal):
			#self.env.reset(use_frite=True)
			# get a random goal = numpy array [x,y,z]
			a_random_goal = self.env.sample_goal_database()
			print("-> {} : Go to GOAL : {} !".format(i,a_random_goal))
			self.env.go_to_position_simulated(a_random_goal)
			print("-> Goal OK !")
			print("->  Save value !")
			time.sleep(self.time_action)
			self.write_floats(f)
			
		print("->Close file !")
		f.close()
			
	def generate_classic(self):
		self.init_spaces()
		self.go_to_corner()
		f = open(self.path_generate + self.generate_name, "w+")
   
		for j in range(self.range_z):
			
			print("plan {} / {}".format(j, self.range_z))
        
			for i in range(math.ceil(self.range_y/2)):
				# advance to x max
				d_x_y_z= [0.0, 0.0, 0.0]
				d_x_y_z[0] = self.step_x
				for x in range(self.range_x):
					self.env.set_action_cartesian(d_x_y_z)
					time.sleep(self.time_action)
					self.env.draw_gripper_position()
					self.write_floats(f)
                        
            
				# 1 shift Y
				d_x_y_z = [0.0, 0.0, 0.0]
				d_x_y_z[1] = self.step_y
				self.env.set_action_cartesian(d_x_y_z)
				time.sleep(self.time_action)
				self.env.draw_gripper_position()
				self.write_floats(f)
        
				# advance to x min
				d_x_y_z= [0.0, 0.0, 0.0]
				d_x_y_z[0] = -self.step_x
				for x in range(self.range_x):
					self.env.set_action_cartesian(d_x_y_z)
					time.sleep(self.time_action)
					self.env.draw_gripper_position()
					self.write_floats(f)
        
				# 1 shift Y
				d_x_y_z = [0.0, 0.0, 0.0]
				d_x_y_z[1] = self.step_y
				self.env.set_action_cartesian(d_x_y_z)
				time.sleep(self.time_action)
				self.env.draw_gripper_position()
				self.write_floats(f)
  
			# 1 shift z
			d_x_y_z= [0.0, 0.0, 0.0]
			d_x_y_z[2] = -self.step_z
        
			self.step_y*=-1
        
		f.close()
		
