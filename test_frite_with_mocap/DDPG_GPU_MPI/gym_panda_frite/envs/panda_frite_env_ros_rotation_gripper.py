import os, inspect
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

import math
from numpy import linalg as LA


import threading
from datetime import datetime
import time

from gym_panda_frite.envs.debug_gui import Debug_Gui


import rospy
import rospkg
from geometry_msgs.msg import PoseArray, Point, Quaternion
from geometry_msgs.msg import PoseStamped

from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

from visualization_msgs.msg import MarkerArray, Marker
	
		
class PandaFriteEnvROSRotationGripper(gym.Env):
	
	def __init__(self, database = None, json_decoder = None, env_pybullet = None, gui = None):
		
		if json_decoder==None:
			raise RuntimeError("=> PandaFriteEnvROS class need a JSON Decoder, to get some parameters !!!")
			return
	
		print("****** PandaFriteEnvROS !!!! ************")
		
		self.json_decoder = json_decoder
		is_ros_version = self.json_decoder.config_data["env"]["is_ros_version"]
		if is_ros_version:
			print("ROS VERSION !!!!!!!!!")
			self.import_ros_packages()
		else:
			print("NO ROS VERSION !!!!!!!!!!!!")
		
		self.gui = gui
		self.database = database
		self.debug_lines_gripper_array = [0, 0, 0, 0]
		
		self.factor_dt_factor = 1.0
		
		self.E = self.json_decoder.config_data["env"]["E"]
		self.NU = self.json_decoder.config_data["env"]["NU"]
		self.add_frite_parameters_to_observation = self.json_decoder.config_data["env"]["add_frite_parameters_to_observation"]
		self.fix_initial_gripper_orientation = self.json_decoder.config_data["env"] ["fix_initial_gripper_orientation"]
		self.initial_gripper_orientation = None
		
		self.dt_factor = self.json_decoder.config_data["env"]["dt_factor"]
		self.joint_motor_control_force = self.json_decoder.config_data["env"]["joint_motor_control_force"]
		self.time_set_action = self.json_decoder.config_data["env"]["time_set_action"]
		self.distance_threshold = self.json_decoder.config_data["env"]["distance_threshold"]
		
		self.do_random_frite = self.json_decoder.config_data["randomization"]["frite"]["do_random"]
		self.min_E = self.json_decoder.config_data["randomization"]["frite"]["E"]["min"]
		self.max_E = self.json_decoder.config_data["randomization"]["frite"]["E"]["max"]
		
		self.E_space = spaces.Box(low=np.array([self.min_E]), high=np.array([self.max_E]))
		
		self.min_NU = self.json_decoder.config_data["randomization"]["frite"]["NU"]["min"]
		self.max_NU = self.json_decoder.config_data["randomization"]["frite"]["NU"]["max"]
		
		self.NU_space = spaces.Box(low=np.array([self.min_NU]), high=np.array([self.max_NU]))
		
		
		# define action space
		
		self.action_x_min = self.json_decoder.config_data["randomization"]["action"]["x"]["min"]
		self.action_x_max = self.json_decoder.config_data["randomization"]["action"]["x"]["max"]
		self.action_y_min = self.json_decoder.config_data["randomization"]["action"]["y"]["min"]
		self.action_y_max = self.json_decoder.config_data["randomization"]["action"]["y"]["max"]
		self.action_z_min = self.json_decoder.config_data["randomization"]["action"]["z"]["min"]
		self.action_z_max = self.json_decoder.config_data["randomization"]["action"]["z"]["max"]
		
		self.action_x_space = spaces.Box(low=np.array([self.action_x_min]), high=np.array([self.action_x_max]))
		self.action_y_space = spaces.Box(low=np.array([self.action_y_min]), high=np.array([self.action_y_max]))
		self.action_z_space = spaces.Box(low=np.array([self.action_z_min]), high=np.array([self.action_z_max]))
		
		# define observation space
		
		self.observation_x_min = self.json_decoder.config_data["randomization"]["observation"]["x"]["min"]
		self.observation_x_max = self.json_decoder.config_data["randomization"]["observation"]["x"]["max"]
		self.observation_y_min = self.json_decoder.config_data["randomization"]["observation"]["y"]["min"]
		self.observation_y_max = self.json_decoder.config_data["randomization"]["observation"]["y"]["max"]
		self.observation_z_min = self.json_decoder.config_data["randomization"]["observation"]["z"]["min"]
		self.observation_z_max = self.json_decoder.config_data["randomization"]["observation"]["z"]["max"]
		
		self.observation_x_space = spaces.Box(low=np.array([self.observation_x_min]), high=np.array([self.observation_x_max]))
		self.observation_y_space = spaces.Box(low=np.array([self.observation_y_min]), high=np.array([self.observation_y_max]))
		self.observation_z_space = spaces.Box(low=np.array([self.observation_z_min]), high=np.array([self.observation_z_max]))
		
		
		# for real
		self.publish_init_pos_mesh = self.json_decoder.config_data["env_test"]["real"]["publish_init_pos_mesh"]
		
		# bullet env parameters + thread time_step
		self.env_pybullet = env_pybullet
		
		self.dt = self.env_pybullet.time_step*self.env_pybullet.n_substeps*self.dt_factor*self.factor_dt_factor
		self.max_vel = 1
		self.max_gripper_vel = 20
		
		self.matrix_base_frame_in_arm_frame = np.array(
											[[1, 0, 0, 0.025],
											[0, 1, 0, 0.422],
											[0, 0, 1, 0.017],
											[0, 0, 0, 1]]
										)
		
		
		self.id_debug_gripper_position = None
		self.id_debug_joints_values = None
		self.id_debug_frite_list = None
		self.id_debug_marker_frite_list = None
		self.id_save_button = None
		
		self.pos_of_mesh_in_obs = None
		self.nb_obs_values = None
		self.nb_action_values = None
		
		# Points (on the front side) from bottom to up
		# [5, 2]
		#   ||
		#   vv
		# [31, 15]
		#   ||
		#   vv
		# [47, 33]
		#   ||
		#   vv
		# [13, 10]
		#   ||
		#   vv
		# [18, 14]
		#   ||
		#   vv
		# [28, 53]
		#   ||
		#   vv
		# [9, 6] (TIP)
		#self.id_frite_to_follow = [ [31, 15], [13, 10], [18, 14], [9, 6] ]  # left then right  [left, right], [left,right] ...
		# -> self.id_frite_to_follow = [ [31, 15], [13, 10], [18, 14], [28, 53] ]  # left then right  [left, right], [left,right] ...
		#self.id_frite_to_follow = [ [31, 15], [47, 33], [18, 14], [28, 53] ]  # left then right  [left, right], [left,right] ...
		
		self.id_frite_to_follow = self.json_decoder.config_data["env"]["id_frite_to_follow"]
		
		# Points from bottom to up, on the same plane of id_frite_to _follow, one level under ((on the front side)
		# [63, 38] (under)
		# [31, 15]
		#   ||
		#   vv
		# [64, 45] (under)
		# [47, 33]
		#   ||
		#   vv
		# [58, 54] (under)
		# [13, 10]
		#   ||
		#   vv
		# [42, 37] (under)
		# [18, 14]
		#   ||
		#   vv
		# [23, 32] (under)
		# [28, 53] (under)
		# [9, 6] (TIP)
		# -> self.under_id_frite_to_follow = [ [63, 38], [58, 54], [42, 37], [23, 32] ]  # left then right  [left, right], [left,right] ...
		#self.under_id_frite_to_follow = [ [63, 38], [64, 45], [42, 37], [23, 32] ]  # left then right  [left, right], [left,right] ...
		
		self.under_id_frite_to_follow = self.json_decoder.config_data["env"]["under_id_frite_to_follow"]
		
		
		# array containing the upper mean point shifted by a normalized normal vector
		self.position_mesh_to_follow = [None, None, None, None]
		
		# array containing the upper mean points (between left and right upper points)
		self.mean_position_to_follow = [None, None, None, None]
		
		# Numpy array that content the Poses and orientation of each rigid body defined with the mocap
		# [[mesh0 geometry_msgs/Pose ], [mesh1 geometry_msgs/Pose] ..., [mesh n geometry_msgs/Pose]]
		self.array_mocap_poses_base_frame = None
		
		#self.debug_id_frite_to_follow = [[None,None],[None,None]]  # draw 2 lines (a cross) per id frite to follow
		
		self.debug_gui = Debug_Gui(env = self)
		
		#print("PandaFriteEnv distance_threshold = {}".format(self.distance_threshold))
		
		self.seed()
		
		# switch tool to convert a numeric value to string value
		self.switcher_type_name = {
			p.JOINT_REVOLUTE: "JOINT_REVOLUTE",
			p.JOINT_PRISMATIC: "JOINT_PRISMATIC",
			p.JOINT_SPHERICAL: "JOINT SPHERICAL",
			p.JOINT_PLANAR: "JOINT PLANAR",
			p.JOINT_FIXED: "JOINT FIXED"
		}
		
		# All Panda Robot joints info
		# => num of joints = 12
		# i=0, name=panda_joint1, type=JOINT_REVOLUTE, lower=-2.8973, upper=2.8973, effort=87.0, velocity=2.175
		# child link name=panda_link1, pos=(0.0, 0.0, 0.333), orien=(0.0, 0.0, 0.0, 1.0)
		# i=1, name=panda_joint2, type=JOINT_REVOLUTE, lower=-1.7628, upper=1.7628, effort=87.0, velocity=2.175
		# child link name=panda_link2, pos=(0.0, 0.0, 0.333), orien=(-0.7071067811848163, 0.0, 0.0, 0.7071067811882787)
		# i=2, name=panda_joint3, type=JOINT_REVOLUTE, lower=-2.8973, upper=2.8973, effort=87.0, velocity=2.175
		# child link name=panda_link3, pos=(0.0, -1.547373340571312e-12, 0.649), orien=(0.0, 0.0, 0.0, 1.0)
		# i=3, name=panda_joint4, type=JOINT_REVOLUTE, lower=-3.0718, upper=-0.0698, effort=87.0, velocity=2.175
		# child link name=panda_link4, pos=(0.08250000000000003, -1.5473802794652158e-12, 0.649), orien=(0.499999999998776, 0.49999999999877587, -0.5000000000012242, 0.5000000000012241)
		# i=4, name=panda_joint5, type=JOINT_REVOLUTE, lower=-2.8973, upper=2.8973, effort=12.0, velocity=2.61
		# child link name=panda_link5, pos=(0.4665000000000001, -1.143409118848924e-12, 0.7315000000000003), orien=(-5.55111512311244e-17, 0.7071067811865476, -3.462452546898476e-12, 0.7071067811865476)
		# i=5, name=panda_joint6, type=JOINT_REVOLUTE, lower=-0.0175, upper=3.7525, effort=12.0, velocity=2.61
		# child link name=panda_link6, pos=(0.4665000000000001, -1.143409118848924e-12, 0.7315000000000003), orien=(0.7071067811848163, 2.7755579854121753e-17, 8.326672684681879e-17, 0.7071067811882789)
		# i=6, name=panda_joint7, type=JOINT_REVOLUTE, lower=-2.8973, upper=2.8973, effort=12.0, velocity=2.61
		# child link name=panda_link7, pos=(0.5545000000000001, -1.1433952410603702e-12, 0.7315000000000003), orien=(0.9238795325112867, -0.3826834323650898, 1.873973198840417e-12, 4.524032781836916e-12)
		# i=7, name=panda_joint8, type=JOINT FIXED, lower=0.0, upper=-1.0, effort=0.0, velocity=0.0
		# child link name=panda_link8, pos=(0.5545000000000001, -2.1913103955728863e-12, 0.6245000000000003), orien=(0.9238795325112867, -0.3826834323650898, 1.873973198840417e-12, 4.524032781836916e-12)
		# i=8, name=panda_hand_joint, type=JOINT FIXED, lower=0.0, upper=-1.0, effort=0.0, velocity=0.0
		# child link name=panda_hand, pos=(0.5545000000000001, -2.1913103955728863e-12, 0.6245000000000003), orien=(1.0, -2.241540286718191e-13, 5.308979892652072e-17, 4.8967997874416646e-12)
		# i=9, name=panda_hand_tcp_joint, type=JOINT FIXED, lower=0.0, upper=-1.0, effort=0.0, velocity=0.0
		# child link name=panda_hand_tcp, pos=(0.5545000000000001, -3.2039685916158226e-12, 0.5211000000000002), orien=(1.0, -2.241540286718191e-13, 5.308979892652072e-17, 4.8967997874416646e-12)
		# i=10, name=panda_finger_joint1, type=JOINT_PRISMATIC, lower=0.0, upper=0.04, effort=100.0, velocity=0.2
		# child link name=panda_leftfinger, pos=(0.5544999999999822, -0.040000000002763256, 0.5661000000003921), orien=(1.0, -2.241540286718191e-13, 5.308979892652072e-17, 4.8967997874416646e-12)
		# i=11, name=panda_finger_joint2, type=JOINT_PRISMATIC, lower=0.0, upper=0.04, effort=100.0, velocity=0.2
		# child link name=panda_rightfinger, pos=(0.5545000000000181, 0.039999999997236746, 0.5660999999996086), orien=(1.0, -2.241540286718191e-13, 5.308979892652072e-17, 4.8967997874416646e-12)

		self.panda_end_eff_idx = 9
		
		self.database.set_env(self)
		self.database.load()
		
		if is_ros_version:
			self.init_ros()
			time.sleep(5)
			self.reset_ros()
		else:
			self.reset_env(use_frite=True)
			self.reset()
			self.panda_list_lower_limits, self.panda_list_upper_limits, self.panda_list_joint_ranges, self.panda_list_initial_poses = self.get_panda_joint_ranges()
			 
		# show sliders
		self.panda_joint_name_to_slider={}
		#self.show_sliders()
		
		#self.show_cartesian_sliders()
		
		#p.stepSimulation()
    
	def import_ros_packages(self):
		import rospy
		import rospkg
		from geometry_msgs.msg import PoseArray, Point, Quaternion
		from geometry_msgs.msg import PoseStamped

		from std_msgs.msg import Float64
		from std_msgs.msg import Float64MultiArray

		from visualization_msgs.msg import MarkerArray, Marker


	def set_E(self, value):
		self.E = float(value)
		
	def set_NU(self, value):
		self.NU = float(value)
		
	def set_factor_dt_factor(self,value):
		self.factor_dt_factor = float(value)
		
	def set_time_step(self, value):
		self.env_pybullet.time_step=float(value)
		self.dt = self.env_pybullet.time_step*self.env_pybullet.n_substeps*self.dt_factor*self.factor_dt_factor
	
	
	def set_initial_gripper_orientation(self, x_rot, y_rot, z_rot):
		self.initial_gripper_orientation = [float(x_rot), float(y_rot), float(z_rot)]
	
		
	def sample_random_action(self):
		return np.array([self.action_x_space.sample(),self.action_y_space.sample(),self.action_z_space.sample()]).flatten()

	def sample_random_observation(self):
		return np.array([self.observation_x_space.sample(),self.observation_y_space.sample(),self.observation_z_space.sample()]).flatten()

	def draw_cross_mocap_mesh(self):
		for i in range(len(self.poses_meshes_in_arm_frame)):
			self.debug_gui.draw_cross("mesh_mocap_" + str(i) , a_pos = [self.poses_meshes_in_arm_frame[i][0],self.poses_meshes_in_arm_frame[i][1],self.poses_meshes_in_arm_frame[i][2]]
		)
	
	def open_database_mocap(self):
		self.file_goal_mocap_poses = open("database_goal_mocap_poses.txt", "w+")
		
	def open_database_mocap_in_read_mode(self):
		self.file_goal_mocap_poses = open("database_goal_mocap_poses.txt")
		
	def close_database_mocap(self):
		self.file_goal_mocap_poses.close()
		
	
	def go_to_position_simulated(self, a_position):
		
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = a_position
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=self.joint_motor_control_force * 240.)
		
	
	def transform_mocap_poses_to_arm_poses(self, mocap_poses, orientation_base_frame_array):
		
		poses_meshes_in_arm_frame = np.array([[None, None, None, None],[None, None, None, None],[None, None, None, None],[None, None, None, None],[None, None, None, None]])
		
		pos_base_frame = mocap_poses[0]
		#orien_base_frame = np.array([orientation_base_frame_array[0], orientation_base_frame_array[1], orientation_base_frame_array[2], orientation_base_frame_array[3]])


		#orien_base_frame = np.array([0.000, 0.000, 0.000, 1.000])
		orien_base_frame = orientation_base_frame_array
		matrix_base_frame_in_mocap_frame = self.to_rt_matrix_numpy(orien_base_frame, pos_base_frame)

		matrix_mocap_frame_in_arm_frame = np.dot(self.matrix_base_frame_in_arm_frame, LA.inv(matrix_base_frame_in_mocap_frame))

		for i in range(5):
			pos_mesh_in_mocap_frame = np.array([mocap_poses[i][0],mocap_poses[i][1],mocap_poses[i][2],1])
			poses_meshes_in_arm_frame[i] = np.dot(matrix_mocap_frame_in_arm_frame, pos_mesh_in_mocap_frame)
		
		return poses_meshes_in_arm_frame
		
	def load_database_mocap(self):
		
		input("Press Enter to start load !")
		self.open_database_mocap_in_read_mode()
	
		poses_mocap_array = np.zeros((5,3))
		nbline = 0
		 
		for line in self.file_goal_mocap_poses.readlines():
			#self.go_to_position_simulated(np.array([0.554, 0.000, 0.521]))
			#time.sleep(3)
			line_split = line.split()
			
			if (len(line_split) != 22):
				print("Erreur on line ", nbline)
				
			else:
				goal_x = float(line_split[0])
				goal_y = float(line_split[1])
				goal_z = float(line_split[2])
				print("goal x = {}, y = {}, z = {} ".format(goal_x, goal_y,goal_z))
				goal_position = np.array([goal_x,goal_y,goal_z])
				
				#self.reset(use_frite=True)
				
				#input("go to goal position !")
				self.go_to_position_simulated(goal_position)
					
				
				gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
				print("gripper pos =", gripper_pos)
				
				# Get orientation x,y,z,w
				orientation_base_frame_array = np.array([float(line_split[3]),float(line_split[4]),float(line_split[5]),float(line_split[6])])
				
				for i in range(5):
					x = float(line_split[((i+1)*3)+4])
					y = float(line_split[((i+1)*3)+5])
					z = float(line_split[((i+1)*3)+6])
					poses_mocap_array[i][0] = x
					poses_mocap_array[i][1] = y
					poses_mocap_array[i][2] = z
						
				poses_mocap_array_in_arm_frame = self.transform_mocap_poses_to_arm_poses(poses_mocap_array, orientation_base_frame_array)
				#print(poses_mocap_array)
				#print(poses_mocap_array_in_arm_frame)
				
				time.sleep(30)
				#input("draw mesh and normal to follow !")
				
				self.compute_mesh_pos_to_follow(draw_normal=True)
				
				#input("draw mesh mocapin arm frame !")
				
				for i in range(len(poses_mocap_array_in_arm_frame)):
					self.debug_gui.draw_cross("mesh_mocap_" + str(i) , a_pos = [poses_mocap_array_in_arm_frame[i][0],poses_mocap_array_in_arm_frame[i][1],poses_mocap_array_in_arm_frame[i][2]])
				
				nbline+=1
				time.sleep(10)
			
		self.close_database_mocap()
		
	def generate_mocap_databases(self):
		
		input("Press Enter to go to Home position !")
		self.go_to_home_position()
		input("Press Enter to start !")
		
		nb_goal_to_sample = 100
		self.open_database_mocap()
		
		for i in range(nb_goal_to_sample):
			a_goal = self.sample_goal_database()
			print("{} -> try goal : {}".format(i,a_goal))
			self.go_to_position(a_goal)
			
			# wait a time to reach the 'goal' position
			time.sleep(10)
			
			self.mutex_array_mocap.acquire()
			try:
				copy_of_array_mocap_poses_base_frame = self.array_mocap_poses_base_frame.copy()
			finally:
				self.mutex_array_mocap.release()
			
			
			base_frame_pose = copy_of_array_mocap_poses_base_frame[0]	
			self.file_goal_mocap_poses.write("{:.3f} {:.3f} {:.3f}".format(a_goal[0], a_goal[1], a_goal[2]))
			self.file_goal_mocap_poses.write(" {:.3f} {:.3f} {:.3f} {:.3f}".format(base_frame_pose.orientation.x, base_frame_pose.orientation.y, base_frame_pose.orientation.z, base_frame_pose.orientation.w))
			for a_pose in copy_of_array_mocap_poses_base_frame:
				self.file_goal_mocap_poses.write(" {:.3f} {:.3f} {:.3f}".format(a_pose.position.x, a_pose.position.y, a_pose.position.z))
			self.file_goal_mocap_poses.write("\n")
			
		self.close_database_mocap()
	
	def sample_goal_database(self):
		# sample a goal np.array[x,y,z] from the goal_space 
		goal = np.array(self.goal_space.sample())
		return goal.copy()
		
		
	def go_to_position(self, a_position, an_orientation = None):
		self.publish_position(a_position,an_orientation)
		
		start=datetime.now()
		
		while True:
			#print("attente !")
			self.mutex_observation.acquire()
			try:
				current_tip_position = self.tip_position.copy()
			finally:
				self.mutex_observation.release()
			
			if np.linalg.norm(current_tip_position - a_position, axis=-1) <= 0.03:
				print("Exit with precision !")
				break
			
			if (datetime.now()-start).total_seconds() >= 2:
				print("Exit with time !")
				break
		
	def go_to_home_position(self):
		# End Eff home position => x=0.554, y=0.000, z=0.521
		self.go_to_position(np.array([0.554, 0.000, 0.521]))
			
	def publish_position(self, command, an_orientation = None):
		pose_msg = PoseStamped()
		pose_msg.pose.position.x = command[0]
		pose_msg.pose.position.y = command[1]
		pose_msg.pose.position.z = command[2]
	
		if (an_orientation == None):
			pose_msg.pose.orientation.x = self.init_cartesian_orientation[0]
			pose_msg.pose.orientation.y = self.init_cartesian_orientation[1]
			pose_msg.pose.orientation.z = self.init_cartesian_orientation[2]
			pose_msg.pose.orientation.w = self.init_cartesian_orientation[3]
		else:
			pose_msg.pose.orientation.x = an_orientation[0]
			pose_msg.pose.orientation.y = an_orientation[1]
			pose_msg.pose.orientation.z = an_orientation[2]
			pose_msg.pose.orientation.w = an_orientation[3]
		
		self.publisher_position.publish(pose_msg)
		
	
	
	def publish_goal(self):
		simple_marker_msg = Marker()
		marker_array_msg = MarkerArray()
		marker_array_msg.markers = []
		
		for i in range(len(self.goal)):
			simple_marker_msg = Marker()
			simple_marker_msg.header.frame_id = "panda_link0"
			simple_marker_msg.header.stamp = rospy.get_rostime()
			simple_marker_msg.ns = "goal_pos_in_arm_frame"
			simple_marker_msg.action = simple_marker_msg.ADD

			simple_marker_msg.type = simple_marker_msg.SPHERE
			simple_marker_msg.scale.x = 0.02
			simple_marker_msg.scale.y = 0.02
			simple_marker_msg.scale.z = 0.02
			simple_marker_msg.color.g = 1.0
			simple_marker_msg.color.a = 1.0
			simple_marker_msg.id = i+50
			
			simple_marker_msg.pose.position.x = self.goal[i][0]
			simple_marker_msg.pose.position.y = self.goal[i][1]
			simple_marker_msg.pose.position.z = self.goal[i][2]
			simple_marker_msg.pose.orientation.x = 0
			simple_marker_msg.pose.orientation.y = 0
			simple_marker_msg.pose.orientation.z = 0
			simple_marker_msg.pose.orientation.w = 1
			
			marker_array_msg.markers.append(simple_marker_msg)
		
		self.publisher_goal_in_arm_frame.publish(marker_array_msg)
		
	
	def publish_initial_mesh_pos(self):
		
		intial_mesh_pos_array = np.array([[0.586, 0.000, -0.225],[0.586, 0.000, 0.032],[0.586, 0.000, 0.288],[0.586, 0.000, 0.481]])
		
		simple_marker_msg = Marker()
		marker_array_msg = MarkerArray()
		marker_array_msg.markers = []
		
		for i in range(len(intial_mesh_pos_array)):
			simple_marker_msg = Marker()
			simple_marker_msg.header.frame_id = "panda_link0"
			simple_marker_msg.header.stamp = rospy.get_rostime()
			simple_marker_msg.ns = "intial_mesh_pos_in_arm_frame"
			simple_marker_msg.action = simple_marker_msg.ADD

			simple_marker_msg.type = simple_marker_msg.SPHERE
			simple_marker_msg.scale.x = 0.02
			simple_marker_msg.scale.y = 0.02
			simple_marker_msg.scale.z = 0.02
			simple_marker_msg.color.g = 1.0
			simple_marker_msg.color.a = 1.0
			simple_marker_msg.id = i+40
			
			simple_marker_msg.pose.position.x = intial_mesh_pos_array[i][0]
			simple_marker_msg.pose.position.y = intial_mesh_pos_array[i][1]
			simple_marker_msg.pose.position.z = intial_mesh_pos_array[i][2]
			simple_marker_msg.pose.orientation.x = 0
			simple_marker_msg.pose.orientation.y = 0
			simple_marker_msg.pose.orientation.z = 0
			simple_marker_msg.pose.orientation.w = 1
			
			marker_array_msg.markers.append(simple_marker_msg)
		
		self.publisher_intial_mesh_pos_in_arm_frame.publish(marker_array_msg)
		
	
	def publish_mocap_mesh(self):
		
		simple_marker_msg = Marker()
		marker_array_msg = MarkerArray()
		marker_array_msg.markers = []
		"""
		line_strip_msg = Marker()
		line_strip_msg.points = []
		
		line_strip_msg.header.frame_id = "panda_link0"
		line_strip_msg.header.stamp = rospy.get_rostime()
		line_strip_msg.id = 31
		line_strip_msg.ns = "points_and_lines_in_arm_frame"
		line_strip_msg.action = line_strip_msg.ADD
		line_strip_msg.type = line_strip_msg.LINE_STRIP
		line_strip_msg.scale.x = 0.01
		line_strip_msg.scale.y = 0.01
		line_strip_msg.scale.z = 0.01
		line_strip_msg.color.b = 1.0
		line_strip_msg.color.a = 1.0
		line_strip_msg.pose.orientation.w = 1.0
		"""

		for i in range(len(self.poses_meshes_in_arm_frame)):
			simple_marker_msg = Marker()
			simple_marker_msg.header.frame_id = "panda_link0"
			simple_marker_msg.header.stamp = rospy.get_rostime()
			simple_marker_msg.ns = "points_and_lines_in_arm_frame"
			simple_marker_msg.action = simple_marker_msg.ADD

			simple_marker_msg.type = simple_marker_msg.SPHERE
			simple_marker_msg.scale.x = 0.02
			simple_marker_msg.scale.y = 0.02
			simple_marker_msg.scale.z = 0.02
			simple_marker_msg.color.b = 1.0
			simple_marker_msg.color.a = 1.0
			simple_marker_msg.id = i
			
			simple_marker_msg.pose.position.x = self.poses_meshes_in_arm_frame[i][0]
			simple_marker_msg.pose.position.y = self.poses_meshes_in_arm_frame[i][1]
			simple_marker_msg.pose.position.z = self.poses_meshes_in_arm_frame[i][2]
			simple_marker_msg.pose.orientation.x = 0
			simple_marker_msg.pose.orientation.y = 0
			simple_marker_msg.pose.orientation.z = 0
			simple_marker_msg.pose.orientation.w = 1
			
			marker_array_msg.markers.append(simple_marker_msg)
			"""
			if i > 0:
				point_msg = Point()
				point_msg.x = self.poses_meshes_in_arm_frame[i][0]
				point_msg.y = self.poses_meshes_in_arm_frame[i][1]
				point_msg.z = self.poses_meshes_in_arm_frame[i][2]
				
				line_strip_msg.points.append(point_msg)
			"""
		
		self.publisher_poses_meshes_in_arm_frame.publish(marker_array_msg)
		"""
		self.publisher_line_strip_in_arm_frame.publish(line_strip_msg)
		"""
		
	def observation_callback(self, msg):
		self.mutex_observation.acquire()
		try:
			
			tip_position_list = [msg.data[0], msg.data[1], msg.data[2]]
			self.tip_position = np.array(tip_position_list)
			
			tip_velocity_list = [msg.data[3], msg.data[4], msg.data[5]]
			self.tip_velocity = np.array(tip_velocity_list)
			
			tip_orientation_list = p.getEulerFromQuaternion([msg.data[6], msg.data[7], msg.data[8], msg.data[9]])
			self.tip_orientation = np.array(tip_orientation_list)
			
			tip_velocity_orientation_list = [msg.data[10], msg.data[11], msg.data[12]]
			self.tip_velocity_orientation = np.array(tip_velocity_orientation_list)
			
			
		finally:
			self.mutex_observation.release()	
		
	def mocap_callback(self, msg):
		
		self.mutex_array_mocap.acquire()
		try:
			self.array_mocap_poses_base_frame = np.array(msg.poses)
		finally:
			self.mutex_array_mocap.release()
		
		pos_base_frame = msg.poses[0].position
		orien_base_frame = msg.poses[0].orientation

		self.matrix_base_frame_in_mocap_frame = self.to_rt_matrix(orien_base_frame, pos_base_frame)

		self.matrix_mocap_frame_in_arm_frame = np.dot(self.matrix_base_frame_in_arm_frame, LA.inv(self.matrix_base_frame_in_mocap_frame))

		self.mutex_array_mocap_in_arm_frame.acquire()
		try:
			for i in range(5):
				pos_mesh_in_mocap_frame = np.array([msg.poses[i].position.x,msg.poses[i].position.y,msg.poses[i].position.z,1])
				self.poses_meshes_in_arm_frame[i] = np.dot(self.matrix_mocap_frame_in_arm_frame, pos_mesh_in_mocap_frame)
		finally:
			self.mutex_array_mocap_in_arm_frame.release()	
		
		self.publish_goal()
				
		#self.draw_cross_mocap_mesh()
		#print(len(self.poses_meshes_in_arm_frame))
		self.publish_mocap_mesh()
		
		if self.publish_init_pos_mesh:
			self.publish_initial_mesh_pos()
		
	def init_ros(self):
		print("INIT ROS !!!!!!!!!!!!")
		rospy.init_node('rl_melodie_node')
		
		self.matrix_base_frame_in_mocap_frame = None
		self.matrix_mocap_frame_in_arm_frame = None
		self.init_cartesian_orientation = np.array([1.000, 0.000, 0.000, 0.000])
										
		# x = -0.0088, y = 0.3064 cm  z = 0.017, 0.3 m								
		self.matrix_base_frame_in_arm_frame = np.array(
											[[1, 0, 0, 0.015],
											[0, 1, 0, 0.34],
											[0, 0, 1, 0.005],
											[0, 0, 0, 1]]
										)
										
		self.poses_meshes_in_arm_frame = np.array([[None, None, None, None],[None, None, None, None],[None, None, None, None],[None, None, None, None],[None, None, None, None]])
		
		self.mutex_observation = threading.Lock()
		self.mutex_array_mocap = threading.Lock()
		
		self.mutex_array_mocap_in_arm_frame = threading.Lock()
		
		# sample a new goal
		self.goal = self.sample_goal()
		
		
		rospy.Subscriber('/PoseAllBodies', PoseArray, self.mocap_callback,
						 queue_size=10)
						 
		rospy.Subscriber('/cartesian_impedance_example_controller/current_observation_orientation', Float64MultiArray, self.observation_callback,
						 queue_size=10)
						 
		self.publisher_poses_meshes_in_arm_frame = rospy.Publisher('/VisualizationPoseArrayMarkersInArmFrame', MarkerArray, queue_size=10)
		
		self.publisher_intial_mesh_pos_in_arm_frame = rospy.Publisher('/VisualizationInitialPoseArrayMarkersInArmFrame', MarkerArray, queue_size=10)
		
		self.publisher_goal_in_arm_frame = rospy.Publisher('/VisualizationGoalArrayMarkersInArmFrame', MarkerArray, queue_size=10)
		
		#self.publisher_line_strip_in_arm_frame = rospy.Publisher('/VisualizationLineStripMarkerInArmFrame', Marker, queue_size=10)
		
		self.publisher_position = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
		
	
	def to_rt_matrix_numpy(self,Q, T):
	
		# Extract the values from Q
		
		qx = Q[0]
		qy = Q[1]
		qz = Q[2]
		qw = Q[3]
		
		d = qx*qx + qy*qy + qz*qz + qw*qw
		s = 2.0 / d
		
		xs = qx * s
		ys = qy * s
		zs = qz * s
		wx = qw * xs
		wy = qw * ys
		wz = qw * zs
		xx = qx * xs
		xy = qx * ys
		xz = qx * zs
		yy = qy * ys
		yz = qy * zs
		zz = qz * zs
		
		
		r00 = 1.0 - (yy + zz)
		r01 = xy - wz
		r02 = xz + wy
		
		r10 = xy + wz
		r11 = 1.0 - (xx + zz)
		r12 = yz - wx
		
		r20 = xz - wy
		r21 = yz + wx
		r22 = 1.0 - (xx + yy)
		
		
		# 4x4 RT matrix
		rt_matrix = np.array([[r00, r01, r02, T[0]],
							   [r10, r11, r12, T[1]],
							   [r20, r21, r22, T[2]],
							   [0, 0, 0, 1]])
								
		return rt_matrix
	
		
	
	def to_rt_matrix(self,Q, T):
	
		# Extract the values from Q
		qw = Q.w
		qx = Q.x
		qy = Q.y
		qz = Q.z
		
		
		d = qx*qx + qy*qy + qz*qz + qw*qw
		s = 2.0 / d
		
		xs = qx * s
		ys = qy * s
		zs = qz * s
		wx = qw * xs
		wy = qw * ys
		wz = qw * zs
		xx = qx * xs
		xy = qx * ys
		xz = qx * zs
		yy = qy * ys
		yz = qy * zs
		zz = qz * zs
		
		
		r00 = 1.0 - (yy + zz)
		r01 = xy - wz
		r02 = xz + wy
		
		r10 = xy + wz
		r11 = 1.0 - (xx + zz)
		r12 = yz - wx
		
		r20 = xz - wy
		r21 = yz + wx
		r22 = 1.0 - (xx + yy)
		
		
		# 4x4 RT matrix
		rt_matrix = np.array([[r00, r01, r02, T.x],
							   [r10, r11, r12, T.y],
							   [r20, r21, r22, T.z],
							   [0, 0, 0, 1]])
								
		return rt_matrix
	
	def frame_transform(self):
		# len(self.position_mesh_to_follow)-1 -> index of TIP -> mesh1
		# world position of point 'tip' shifted by a normal vector to follow  (TIP)
		mesh1_pos_world = self.position_mesh_to_follow[len(self.position_mesh_to_follow)-1]
		
		# 4 vector of shifted_tip_pos_world
		mesh1_pos_world_v4 = np.array([mesh1_pos_world[0], mesh1_pos_world[1], mesh1_pos_world[2], 1])
		
		# end effector (tip/gripper) position and orientation (quaternion)
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		gripper_orient = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[1]
		
		# matrix that express the tip/gripper pos into world frame
		world_gripper_rt_matrix = self.to_rt_matrix(gripper_orient,gripper_pos)
		
		# LA.inv(rt_matrix) -> tip world matrix RT   
		# ( Mworld->gripper = gripper express into world frame)
		# LA.inv(world_gripper_rt_matrix) -> Mgripper->world  ( world express into tip/gripper frame)
		# Mgripper->world * mesh1_pos_world_v4 (mesh1 express into world frame) -> mesh1 express into tip/gripper frame
		
		mesh1_in_tip_frame = np.dot(LA.inv(world_gripper_rt_matrix),mesh1_pos_world_v4)
		
		# mesh1 norm express into tip frame
		mesh1_norm = LA.norm(mesh1_in_tip_frame[0:3])
		print(mesh1_norm)
		
		print("dx={}, dy={}, dz={}".format((mesh1_pos_world[0]-gripper_pos[0]),(mesh1_pos_world[1]-gripper_pos[1]), (mesh1_pos_world[2]-gripper_pos[2]) ))
		print("x={}, y={}, z={}".format(mesh1_in_tip_frame[0], mesh1_in_tip_frame[1], mesh1_in_tip_frame[2]))
		
		
	# Shift 'pt_mean' point with a distance 'a_distance' 
	# using a normal vector calculated from 3 points : 'pt_left', 'pt_right', 'pt_mean'. 
	def shift_point_in_normal_direction(self, pt_left, pt_right, pt_mean, a_distance = 0.2):
		# under left and right points
		vleft = np.array([pt_left[0], pt_left[1], pt_left[2]])
		vright = np.array([pt_right[0], pt_right[1], pt_right[2]])
		
		# upper mean point
		vmean = np.array([pt_mean[0], pt_mean[1], pt_mean[2]])
		
		# calculate the normal vector using the cross product of two (arrays of) vectors.
		vnormal = np.cross(vleft-vmean, vright-vmean)
		
		# calculate the norm of the normal vector
		norm_of_vnormal = np.linalg.norm(vnormal)
		
		# Normalize the normal vector 
		vnormal_normalized = vnormal / norm_of_vnormal
		
		# Shift the upper mean point of a distance by using the normal vector normalized 
		vmean_shifted = vmean + vnormal_normalized * a_distance
		
		return vmean_shifted
	
	
	def draw_frite_parameters(self):
		str_to_print = "E=" + str(self.E) + ", NU=" + str(self.NU) + ", timeStep=" + str(self.env_pybullet.time_step) + ", factor_dt_factor=" + str(self.factor_dt_factor)
		if self.database.type_db_load == 2 and self.fix_initial_gripper_orientation:
			str_to_print += ", orien=" + str(self.initial_gripper_orientation)
		self.debug_gui.draw_text("frite_parameters", a_text=str_to_print, a_pos=[1,1,1], a_size=1.0)
		
		
	def draw_id_to_follow(self):
		for i in range(len(self.id_frite_to_follow)):
			self.debug_gui.draw_cross("id_frite_"+str(i), a_pos = self.position_mesh_to_follow[i], a_color = [0, 0, 1])
			#self.debug_gui.draw_text("text_id_frite_"+str(i), a_text = str(i), a_pos = self.position_mesh_to_follow[i], a_color = [0, 0, 1])

	def draw_normal_plane(self, index, data, a_normal_pt):
		# self.id_frite_to_follow[index][0] -> upper left
		# self.id_frite_to_follow[index][1] -> upper right
		# self.under_id_frite_to_follow[index][0] -> under left
		# self.under_id_frite_to_follow[index][1] -> under right
		
		# Draw a square by using upper (left/right) and under (left/right) points
		self.debug_gui.draw_line(name="l_"+str(index)+"_up",a_pos_from = data[1][self.id_frite_to_follow[index][0]], a_pos_to = data[1][self.id_frite_to_follow[index][1]])
		self.debug_gui.draw_line(name="l_"+str(index)+"_bottom",a_pos_from = data[1][self.under_id_frite_to_follow[index][0]], a_pos_to = data[1][self.under_id_frite_to_follow[index][1]])
		self.debug_gui.draw_line(name="l_"+str(index)+"_left",a_pos_from = data[1][self.id_frite_to_follow[index][0]], a_pos_to = data[1][self.under_id_frite_to_follow[index][0]])
		self.debug_gui.draw_line(name="l "+str(index)+"_right",a_pos_from = data[1][self.id_frite_to_follow[index][1]], a_pos_to = data[1][self.under_id_frite_to_follow[index][1]])
		
		# Draw a line for the normal vector from the mean upper point
		self.debug_gui.draw_line(name="normal_"+str(index),a_pos_from = self.mean_position_to_follow[index], a_pos_to = a_normal_pt, a_color = [1, 1, 0])
		
	def compute_mesh_pos_to_follow(self, draw_normal=False):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		
		
		# For all id to follow except the TIP
		for i in range(len(self.id_frite_to_follow)):
			
			# get left and right upper points 
			a_pt_left = np.array(data[1][self.id_frite_to_follow[i][0]])
			a_pt_right = np.array(data[1][self.id_frite_to_follow[i][1]])
			
			# calculate the upper mean (front side) between left and right upper points
			self.mean_position_to_follow[i] = (a_pt_left + a_pt_right)/2.0
			
			# get left and right under points
			a_pt_left_under = np.array(data[1][self.under_id_frite_to_follow[i][0]])
			a_pt_right_under = np.array(data[1][self.under_id_frite_to_follow[i][1]])
			
			# calculate the upper mean point shifted by a normalized normal vector.
			# The normal vector is calculated from a triangle defined by left+right under points and upper mean point.
			# 0.007 is equal to the half of the marker thickness
			self.position_mesh_to_follow[i] = self.shift_point_in_normal_direction(pt_left=a_pt_left_under, pt_right=a_pt_right_under, pt_mean=self.mean_position_to_follow[i], a_distance = 0.007)
			
			if draw_normal:
				a_normal_pt = self.shift_point_in_normal_direction(pt_left=a_pt_left_under, pt_right=a_pt_right_under, pt_mean=self.mean_position_to_follow[i], a_distance = 0.1)
				self.draw_normal_plane(i, data, a_normal_pt)
				
	def draw_cross_mesh_to_follow(self):
		for i in range(len(self.position_mesh_to_follow)):
			self.debug_gui.draw_cross("mesh_frite_" + str(i) , a_pos = self.position_mesh_to_follow[i])
	
	def compute_height_id_frite(self):
		#self.id_frite_to_follow = [ [31, 15], [47, 33], [18, 14], [28, 53] ]
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		pos_2 = data[1][2]
		pos_15 = data[1][15]
		pos_33 = data[1][33]
		pos_10 = data[1][10]
		pos_14 = data[1][14]
		pos_53 = data[1][53]
		
		
		z_diff_2_15 = pos_2[2] - pos_15[2]
		z_diff_2_33 = pos_2[2] - pos_33[2]
		z_diff_2_10 = pos_2[2] - pos_10[2]
		z_diff_2_14 = pos_2[2] - pos_14[2]
		z_diff_2_53 = pos_2[2] - pos_53[2]
		
		self.debug_gui.draw_text("z_diff_2_15", a_text = "z_diff_2_15=" + str(z_diff_2_15), a_pos=[1,1,1])
		self.debug_gui.draw_text("z_diff_2_33", a_text = "z_diff_2_33=" + str(z_diff_2_33), a_pos=[1,1,1.5])
		self.debug_gui.draw_text("z_diff_2_10", a_text = "z_diff_2_10=" + str(z_diff_2_10), a_pos=[1,1,2.0])
		self.debug_gui.draw_text("z_diff_2_14", a_text = "z_diff_2_14=" + str(z_diff_2_14), a_pos=[1,1,2.5])
		self.debug_gui.draw_text("z_diff_2_53", a_text = "z_diff_2_53=" + str(z_diff_2_53), a_pos=[1,1,3.0])
		
		"""
		z_diff_2_15 = pos_2[2] - pos_15[2]
		z_diff_15_33 = pos_15[2] - pos_33[2]
		z_diff_33_10 = pos_33[2] - pos_10[2]
		z_diff_10_14 = pos_10[2] - pos_14[2]
		z_diff_14_53 = pos_14[2] - pos_53[2]
		
		self.debug_gui.draw_text("z_diff_2_15", a_text = "z_diff_2_15=" + str(z_diff_2_15), a_pos=[1,1,1])
		self.debug_gui.draw_text("z_diff_15_33", a_text = "z_diff_15_33=" + str(z_diff_15_33), a_pos=[1,1,1.5])
		self.debug_gui.draw_text("z_diff_33_10", a_text = "z_diff_33_10=" + str(z_diff_33_10), a_pos=[1,1,2.0])
		self.debug_gui.draw_text("z_diff_10_14", a_text = "z_diff_10_14=" + str(z_diff_10_14), a_pos=[1,1,2.5])
		self.debug_gui.draw_text("z_diff_14_53", a_text = "z_diff_14_53=" + str(z_diff_14_53), a_pos=[1,1,3.0])
		"""
		
		
	def draw_all_ids_mesh_frite(self):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		flat_array = np.array(self.id_frite_to_follow).flatten()
		
		for i in range(data[0]):
			pos = data[1][i]
			if i in flat_array:
				p.addUserDebugText(str(i), pos, textColorRGB=[1,0,0])
			else:
				p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
			
	def draw_ids_mesh_frite(self, a_from=0, a_to=0):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		flat_array = np.array(self.id_frite_to_follow).flatten()
		
		for i in range(data[0]):
			if (i>=a_from and i<=a_to):
				pos = data[1][i]
				if i in flat_array:
					p.addUserDebugText(str(i), pos, textColorRGB=[1,0,0])
				else:
					p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
	
	def get_gripper_position(self):
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		return gripper_pos
		
	def draw_text_joints_values(self):
		joints_values=[]
		for joint_name in self.panda_joint_name_to_ids.keys():
			if (joint_name != 'panda_finger_joint1' and joint_name != 'panda_finger_joint2'):
				joint_index = self.panda_joint_name_to_ids[joint_name]
				joints_values.append(p.getJointState(self.panda_id, joint_index)[0])
				
		str_joints_values = ""
		for i in range(len(joints_values)):
			str_joints_values+="q{}={:.3f}, ".format(i+1,joints_values[i])
			
		self.debug_gui.draw_text(a_name="joint_values" , a_text=str_joints_values, a_pos = [-1.0,0,0.5])

	def draw_text_gripper_position(self):
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		gripper_pos_str = "x={:.3f}, y={:.3f}, z={:.3f}".format(gripper_pos[0], gripper_pos[1], gripper_pos[2])
		self.debug_gui.draw_text(a_name="gripper_pos" , a_text=gripper_pos_str, a_pos = [-1.0,0,0.1])


	def draw_gripper_position(self):
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		
		self.debug_gui.draw_cross("gripper_pos" , a_pos = gripper_pos)
		
	def draw_env_box(self):
		self.debug_gui.draw_box(self.pos_space.low, self.pos_space.high, [0, 0, 1])
		self.debug_gui.draw_box(self.goal_space.low, self.goal_space.high, [1, 0, 0])

	def draw_goal(self):
		for i in range(self.goal.shape[0]):
			#print("draw_goal[{}]={}".format(i,self.goal[i]))
			self.debug_gui.draw_cross("goal_"+str(i) , a_pos = self.goal[i])
			#self.debug_gui.draw_text("text_goal_"+str(i), a_text = str(i), a_pos = self.goal[i])
			
			
	def draw_gripper_position(self):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		self.debug_gui.draw_cross("gripper", a_pos = cur_pos, a_color = [0, 0, 1])
	
	def set_env_with_frite_parameters(self, values_frite_parameters):
		E = values_frite_parameters[0]
		NU = values_frite_parameters[1]
		time_step = values_frite_parameters[2]
		factor_dt_factor = values_frite_parameters[3]
		x_rot = values_frite_parameters[4]
		y_rot = values_frite_parameters[5]
		z_rot = values_frite_parameters[6]
		
		
		self.set_E(E)
		self.set_NU(NU)
		self.set_factor_dt_factor(factor_dt_factor)
		self.set_time_step(time_step)
		self.set_initial_gripper_orientation(x_rot, y_rot, z_rot)
		
		
	def sample_goal(self):
		if self.database.type_db_load == 2:
			# db random with frite parameters
			goal_with_frite_parameters = self.database.get_random_targets_with_frite_parameters()
			self.set_env_with_frite_parameters(goal_with_frite_parameters[0])
			#print("goal_with_frite_parameters = {}".format(goal_with_frite_parameters))
			goal = goal_with_frite_parameters[1]
		else:
			goal = self.database.get_random_targets()
		
		return goal
	
	
	def show_cartesian_sliders(self):
		self.list_slider_cartesian = []
		self.list_slider_cartesian.append(p.addUserDebugParameter("VX", -1, 1, 0))
		self.list_slider_cartesian.append(p.addUserDebugParameter("VY", -1, 1, 0))
		self.list_slider_cartesian.append(p.addUserDebugParameter("VZ", -1, 1, 0))
		#self.list_slider_cartesian.append(p.addUserDebugParameter("Theta_Dot", -1, 1, 0))
				
	
	def apply_cartesian_sliders(self):
		action = np.empty(3, dtype=np.float64)
		
		for i in range(3):
			action[i] = p.readUserDebugParameter(self.list_slider_cartesian[i])
		
		self.set_action(action)
		#p.stepSimulation()


	def set_gym_spaces_ros(self):
		
		self.mutex_observation.acquire()
		try:
			
			tip_position = self.tip_position.copy()
			
		finally:
			self.mutex_observation.release()
		
		goal_index = self.json_decoder.config_data["env"]["gym_spaces"]["goal"]
		
		goal_x_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["x_up"]
		goal_x_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["x_down"]
		goal_y_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["y_up"]
		goal_y_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["y_down"]
		goal_z_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["z_down"]
		
		low_x_down = tip_position[0]-goal_x_down
		low_x_up = tip_position[0]+goal_x_up
		
		low_y_down = tip_position[1]-goal_y_down
		low_y_up = tip_position[1]+goal_y_up
		
		low_z_down = tip_position[2]-goal_z_down
		low_z_up = tip_position[2]
		
		self.goal_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		
		pose_index = self.json_decoder.config_data["env"]["gym_spaces"]["pose"]
		
		pose_x_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["x_up"]
		pose_x_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["x_down"]
		pose_y_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["y_up"]
		pose_y_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["y_down"]
		pose_z_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["z_down"]
		
		low_x_down = tip_position[0]-pose_x_down
		low_x_up = tip_position[0]+pose_x_up
		
		low_y_down = tip_position[1]-pose_y_down
		low_y_up = tip_position[1]+pose_y_up
		
		
		low_z_down = tip_position[2]-pose_z_down
		low_z_up = tip_position[2]
		
		self.pos_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		
		# action_space = cartesian world velocity (vx, vy, vz, v_theta_x, v_theta_y, v_theta_z)  = 6 float
		self.action_space = spaces.Box(-1., 1., shape=(6,), dtype=np.float32)
		
		# observation = 36 float -> see function _get_obs
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(36,), dtype=np.float32)

	
	def set_gym_spaces(self):
		panda_eff_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		
		"""
		# LARGE
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-4*low_marge
		low_y_up = panda_eff_state[0][1]+4*low_marge
		
		
		z_low_marge = 0.3
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		
		"""
		# MEDIUM
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-3*low_marge
		low_y_up = panda_eff_state[0][1]+3*low_marge
		
		
		z_low_marge = 0.25
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		
		"""
		# EXTRA SMALL
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-0.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.25*low_marge
		
		low_y_down = panda_eff_state[0][1]-1.5*low_marge
		low_y_up = panda_eff_state[0][1]+1.5*low_marge
		
		
		#z_low_marge = 0.25
		z_low_marge = 0.10
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		
		"""
		# SMALL
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.0*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-2.5*low_marge
		low_y_up = panda_eff_state[0][1]+2.5*low_marge
		
		
		z_low_marge = 0.25
		#z_low_marge = 0.10
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		
		self.goal_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		#print("frite env goal space = {}".format(self.goal_space))
		"""
		
		"""
		# POSE LARGE
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-2*low_marge
		low_x_up = panda_eff_state[0][0]+low_marge

		low_y_down = panda_eff_state[0][1]-5*low_marge
		low_y_up = panda_eff_state[0][1]+5*low_marge

		z_low_marge = 0.3
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		
		"""
		
		"""
		# POSE SMALL
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.0*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-2.5*low_marge
		low_y_up = panda_eff_state[0][1]+2.5*low_marge
		
		
		z_low_marge = 0.25
		#z_low_marge = 0.10
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
	
		"""
		# POSE MEDIUM
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.75*low_marge
		
		low_y_down = panda_eff_state[0][1]-3*low_marge
		low_y_up = panda_eff_state[0][1]+3*low_marge
		
		
		z_low_marge = 0.3
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		
		goal_index = self.json_decoder.config_data["env"]["gym_spaces"]["goal"]
		
		goal_x_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["x_up"]
		goal_x_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["x_down"]
		goal_y_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["y_up"]
		goal_y_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["y_down"]
		goal_z_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][goal_index]["z_down"]
		
		#print("goal_x_up={},goal_x_down={},goal_y_up={},goal_y_down={},goal_z_down={}".format(goal_x_up,goal_x_down,goal_y_up,goal_y_down,goal_z_down))
		
		
		low_x_down = panda_eff_state[0][0]-goal_x_down
		low_x_up = panda_eff_state[0][0]+goal_x_up
		
		low_y_down = panda_eff_state[0][1]-goal_y_down
		low_y_up = panda_eff_state[0][1]+goal_y_up
		
		
		low_z_down = panda_eff_state[0][2]-goal_z_down
		low_z_up = panda_eff_state[0][2]
		
		self.goal_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		
		
		pose_index = self.json_decoder.config_data["env"]["gym_spaces"]["pose"]
		
		pose_x_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["x_up"]
		pose_x_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["x_down"]
		pose_y_up = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["y_up"]
		pose_y_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["y_down"]
		pose_z_down = self.json_decoder.config_data["env"]["gym_spaces"]["spaces"][pose_index]["z_down"]
		
		
		#print("pose_x_up={},pose_x_down={},pose_y_up={},pose_y_down={},pose_z_down={}".format(pose_x_up,pose_x_down,pose_y_up,pose_y_down,pose_z_down))
		
		
		low_x_down = panda_eff_state[0][0]-pose_x_down
		low_x_up = panda_eff_state[0][0]+pose_x_up
		
		low_y_down = panda_eff_state[0][1]-pose_y_down
		low_y_up = panda_eff_state[0][1]+pose_y_up
		
		
		low_z_down = panda_eff_state[0][2]-pose_z_down
		low_z_up = panda_eff_state[0][2]
		
		self.pos_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		
		self.nb_action_values = 6
		
		if self.database.type_db_load == 2 and self.fix_initial_gripper_orientation:
			self.nb_action_values -= 3
		
		print("-> NB Action values = {}".format(self.nb_action_values))
		
		# action_space = cartesian world velocity (vx, vy, vz) + orientation gripper ((thetaX, thetaY, thetaZ) if necessary)
		self.action_space = spaces.Box(-1., 1., shape=(self.nb_action_values,), dtype=np.float32)
		
		self.nb_obs_values = 0
		
		# add gripper link cartesian world position (x,y,z)
		self.nb_obs_values += 3
		
		# add gripper link cartesian world velocity (vx, vy, vz)
		self.nb_obs_values += 3
		
		# add gripper link cartesian world orientation (theta_x,theta_y,theta_z)
		self.nb_obs_values += 3

		# add gripper link cartesian world angular velocity (theta__dot_x,theta__dot_y,theta__dot_z)
		self.nb_obs_values += 3
		
		# add current cartesian world position of id frite to follow ((x,y,z) x 4)
		self.nb_obs_values += len(self.id_frite_to_follow) * 3
		
		# add goal cartesian world position of id frite to reach ((x,y,z) x 4)
		self.nb_obs_values += len(self.id_frite_to_follow) * 3
		
		self.pos_of_mesh_in_obs = 12
		
		if self.database.type_db_load == 2:
			# if type is frite parameters
			
			if self.fix_initial_gripper_orientation:
				# remove gripper link world orientation and angular velocity
				self.nb_obs_values -= 6
				self.pos_of_mesh_in_obs = 6
				
			if self.add_frite_parameters_to_observation:
				# add E and NU
				self.nb_obs_values += 2
			
		print("-> NB Observation values = {}".format(self.nb_obs_values))
		 	
		# observation = X float -> see function _get_obs 
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(self.nb_obs_values,), dtype=np.float32)
			
	
	def get_position_id_frite(self):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		
		pos_list = []

		for i in range(len(self.id_frite_to_follow)):
			pos_list.append(data[1][self.id_frite_to_follow[i]])
		
		return pos_list
			
	def get_panda_joint_ranges(self):
		list_lower_limits, list_upper_limits, list_joint_ranges, list_initial_poses = [], [], [], []

		for item in self.panda_joint_name_to_ids.items():
			joint_name = item[0]
			joint_index = item[1] 
			joint_info = p.getJointInfo(self.panda_id, joint_index)

			a_lower_limit, an_upper_limit = joint_info[8:10]
			#print(item, ' ', a_lower_limit, ' ' , an_upper_limit)

			a_range = an_upper_limit - a_lower_limit

			# For simplicity, assume resting state = initial position
			an_inital_pose = self.panda_initial_positions[joint_name]

			list_lower_limits.append(a_lower_limit)
			list_upper_limits.append(an_upper_limit)
			list_joint_ranges.append(a_range)
			list_initial_poses.append(an_inital_pose)

		return list_lower_limits, list_upper_limits, list_joint_ranges, list_initial_poses
	
	def printPandaAllInfo(self):
		print("=================================")
		print("All Panda Robot joints info")
		num_joints = p.getNumJoints(self.panda_id)
		print("=> num of joints = {0}".format(num_joints))
		for i in range(num_joints):
			joint_info = p.getJointInfo(self.panda_id, i)
			#print(joint_info)
			joint_name = joint_info[1].decode("UTF-8")
			joint_type = joint_info[2]
			child_link_name = joint_info[12].decode("UTF-8")
			link_pos_in_parent_frame = p.getLinkState(self.panda_id, i)[0]
			link_orien_in_parent_frame = p.getLinkState(self.panda_id, i)[1]
			joint_type_name = self.switcher_type_name.get(joint_type,"Invalid type")
			joint_lower_limit, joint_upper_limit = joint_info[8:10]
			joint_limit_effort = joint_info[10]
			joint_limit_velocity = joint_info[11]
			print("i={0}, name={1}, type={2}, lower={3}, upper={4}, effort={5}, velocity={6}".format(i,joint_name,joint_type_name,joint_lower_limit,joint_upper_limit,joint_limit_effort,joint_limit_velocity))
			print("child link name={0}, pos={1}, orien={2}".format(child_link_name,link_pos_in_parent_frame,link_orien_in_parent_frame))
		print("=================================")


	def conv_module_d_young_to_lame(self, E, NU):
		a_lambda = (E * NU)/((1+NU)*(1-2*NU))
		a_mu = E/(2*(1+NU))
		
		return (a_lambda,a_mu)
		
	
	
	def set_panda_to_initial_gripper_orientation(self):
		
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		cur_orien_euler = p.getEulerFromQuaternion(cur_orien)
		if self.initial_gripper_orientation is not None:
			cur_orien_euler = (cur_orien_euler[0] + float(self.initial_gripper_orientation[0]), cur_orien_euler[1] + float(self.initial_gripper_orientation[1]), cur_orien_euler[2] + float(self.initial_gripper_orientation[2]))
		
		new_orien_quaternion = p.getQuaternionFromEuler(cur_orien_euler)
			
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, cur_pos, new_orien_quaternion)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=self.joint_motor_control_force * 240.)
			
		
	def load_frite(self):
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		self.frite_startOrientation = p.getQuaternionFromEuler([0,0,math.pi/4])

		frite_z_position = self.plane_height + self.cube_height
		self.frite_startPos = [gripper_pos[0], gripper_pos[1], frite_z_position]
		
		"""
		self.debug_gui.draw_cross("frite_bottom" , a_pos = self.frite_startPos)
		
		
		frite_up_pos = [gripper_pos[0], gripper_pos[1], frite_z_position + 1.03]
		self.debug_gui.draw_cross("frite_up" , a_pos = frite_up_pos)
		"""
		
		# plage E -> 0.1 à 40
		# frite blanche :
		# E = 0.1*pow(10,6)  NU = 0.49
		
		# frite noire :  E = 35 , NU = 0.46
		# E = 40*pow(10,6)  NU = 0.49
		
		if (self.do_random_frite):
			E = self.E_space.sample()*pow(10,6)
			NU = self.NU_space.sample()
			print("Use Random E={}, NU={}".format(E,NU))
		else:
			E = self.E*pow(10,6)
			NU = self.NU
			print("Use JSON E={}, NU={}".format(E,NU))
			
		(a_lambda,a_mu) = self.conv_module_d_young_to_lame(E,NU)
		
		#print("frite a_lambda={}, a_mu={}".format(a_lambda,a_mu))
		
		
		vtk_file_name = self.json_decoder.config_dir_name + self.json_decoder.config_data["env"]["vtk_file_name"]
		
		# frite : 103 cm with 0.1 cell size
		self.frite_id = p.loadSoftBody(vtk_file_name, basePosition = self.frite_startPos, baseOrientation=self.frite_startOrientation, mass = 0.2, useNeoHookean = 1, NeoHookeanMu = a_mu, NeoHookeanLambda = a_lambda, NeoHookeanDamping = 0.01, useSelfCollision = 1, collisionMargin = 0.001, frictionCoeff = 0.5, scale=1.0)
		#p.changeVisualShape(self.frite_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
			
		"""
		self.frite_id = p.loadSoftBody(
                fileName="vtk/frite.vtk",
                basePosition=self.frite_startPos,
                baseOrientation=self.frite_startOrientation,
                collisionMargin=0.001,
                scale=1.0,
                mass=0.2,
                useNeoHookean=0,
                useBendingSprings=1,
                useMassSpring=1,
                springElasticStiffness=240,
                springDampingStiffness=10,
                springDampingAllDirections=0,
                useSelfCollision=1,
                frictionCoeff=1.0,
                useFaceContact=1,)
		"""
	def load_plane(self):
		self.plane_height = -0.85
		self.plane_id = p.loadURDF("urdf/plane.urdf", basePosition=[0,0,self.plane_height], useFixedBase=True)
		
	def load_cube(self):
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		self.cube_height = 0.365
		cube_z_position = self.plane_height + (self.cube_height / 2.0)
		# load cube
		self.cube_startPos = [gripper_pos[0], gripper_pos[1], cube_z_position]
		self.cube_id = p.loadURDF("urdf/my_cube.urdf", self.cube_startPos, useFixedBase=True)
		

	def load_panda(self):
		self.panda_startOrientation = p.getQuaternionFromEuler([0,0,0])
		# load panda
		self.panda_startPos = [0.0, 0.0, 0.0]

		self.panda_id = p.loadURDF("urdf/franka_panda/panda.urdf",
								   basePosition=self.panda_startPos, baseOrientation=self.panda_startOrientation, useFixedBase=True)

		self.panda_num_joints = p.getNumJoints(self.panda_id) # 12 joints

		#print("panda num joints = {}".format(self.panda_num_joints))
		
		

	def set_panda_initial_joints_positions(self, init_gripper = True):
		self.panda_joint_name_to_ids = {}
		
		# Set intial positions 
		self.panda_initial_positions = {
		'panda_joint1': 0.0, 'panda_joint2': 0.0, 'panda_joint3': 0.0,
		'panda_joint4': -math.pi/2., 'panda_joint5': 0.0, 'panda_joint6': math.pi/2,
		'panda_joint7': math.pi/4., 'panda_finger_joint1': 0.04, 'panda_finger_joint2': 0.04,
		} 
		
		for i in range(self.panda_num_joints):
			joint_info = p.getJointInfo(self.panda_id, i)
			joint_name = joint_info[1].decode("UTF-8")
			joint_type = joint_info[2]
			
			if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
				assert joint_name in self.panda_initial_positions.keys()

				joint_type_name = self.switcher_type_name.get(joint_type,"Invalid type")
                    
				#print("joint {0}, type:{1} -> {2}".format(joint_name,joint_type,joint_type_name))
				self.panda_joint_name_to_ids[joint_name] = i
				
				if joint_name == 'panda_finger_joint1' or joint_name == 'panda_finger_joint2':
					if init_gripper:
						p.resetJointState(self.panda_id, i, self.panda_initial_positions[joint_name])
				else:
					p.resetJointState(self.panda_id, i, self.panda_initial_positions[joint_name])


	def show_sliders(self, prefix_name = '', joint_values=None):
		index = 0
		for item in self.panda_joint_name_to_ids.items():
			joint_name = item[0]
			if (joint_name != 'panda_finger_joint1' and joint_name != 'panda_finger_joint2'):
				joint_index = item[1]
				ll = self.panda_list_lower_limits[index]
				ul = self.panda_list_upper_limits[index]
				if joint_values != None:
					joint_value = joint_values[index]
				else:   
					joint_value = self.panda_initial_positions[joint_name]
				slider = p.addUserDebugParameter(prefix_name + joint_name, ll, ul, joint_value) # add a slider for that joint with the limits
				self.panda_joint_name_to_slider[joint_name] = slider
			index = index + 1 

	def apply_sliders(self):
		for joint_name in self.panda_joint_name_to_ids.keys():
			if (joint_name != 'panda_finger_joint1' and joint_name != 'panda_finger_joint2'):
				slider = self.panda_joint_name_to_slider[joint_name] # get the slider of that joint name
				slider_value = p.readUserDebugParameter(slider) # read the slider value
				joint_index = self.panda_joint_name_to_ids[joint_name]
				p.setJointMotorControl2(self.panda_id, joint_index, p.POSITION_CONTROL,
											targetPosition=slider_value,
											positionGain=0.2)
				#p.stepSimulation()


	def close_gripper(self):
		id_finger_joint1  = self.panda_joint_name_to_ids['panda_finger_joint1']
		id_finger_joint2  = self.panda_joint_name_to_ids['panda_finger_joint2']

		p.setJointMotorControl2(self.panda_id, id_finger_joint1, 
								p.POSITION_CONTROL,targetPosition=0.025,
								positionGain=1.0)
								
		p.setJointMotorControl2(self.panda_id, id_finger_joint2 , 
								p.POSITION_CONTROL,targetPosition=0.025,
								positionGain=1.0)
								
    
	def create_anchor_panda(self):
		
		# mesh id = 0 -> down
		# -1, -1 -> means anchor to the cube
		p.createSoftBodyAnchor(self.frite_id, 0, self.cube_id , -1, [0,0,0])
		p.createSoftBodyAnchor(self.frite_id, 5, self.cube_id , -1, [0,0,0])
		p.createSoftBodyAnchor(self.frite_id, 2, self.cube_id , -1, [0,0,0])
		
		p.createSoftBodyAnchor(self.frite_id, 4, self.cube_id , -1, [0,0,0])
		p.createSoftBodyAnchor(self.frite_id, 3, self.cube_id , -1, [0,0,0])
		#p.createSoftBodyAnchor(self.frite_id, 0, -1 , -1)
		
		# mesh id = 1 -> up
		p.createSoftBodyAnchor(self.frite_id, 1, self.panda_id , self.panda_end_eff_idx, [0,0,0])
		
		# panda finger joint 1 = 10
		# panda finger joint 2 = 11
		"""pos_10 = p.getLinkState(self.panda_id, 10)[0]
		new_pos_10  = [pos_10[0]+0.025, pos_10[1]-0.01, pos_10[2]-0.02]
		pos_11 = p.getLinkState(self.panda_id, 11)[0]
		new_pos_11  = [pos_11[0]+0.025, pos_11[1]+0.01, pos_11[2]-0.02]"""
		p.createSoftBodyAnchor(self.frite_id, 6, self.panda_id , 10, [0.025,-0.01,-0.02])
		p.createSoftBodyAnchor(self.frite_id, 9, self.panda_id , 11, [0.025,0.01,-0.02])
		
		#p.createSoftBodyAnchor(self.frite_id, 6, self.panda_id , 10, [0.025,0.056,-0.02])
		#p.createSoftBodyAnchor(self.frite_id, 9, self.panda_id , 11, [0.025,-0.056,-0.02])
		
		
		
	def set_action_cartesian(self, action):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = cur_pos + np.array(action)
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=self.joint_motor_control_force * 240.)
			
			
	def set_action_ros(self, action):
		assert action.shape == (6,), 'action shape error'
		
		self.mutex_observation.acquire()
		try:
			
			tip_position = self.tip_position.copy()
			
			tip_orientation = self.tip_orientation.copy()
			
		finally:
			self.mutex_observation.release()
			
		new_pos = tip_position + np.array(action[:3]) * self.max_vel * self.dt
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		
		new_orien_euler = tip_orientation + np.array(action[3:]) * self.max_vel * self.dt
		new_orien_quaternion = p.getQuaternionFromEuler(new_orien_euler)
		
		self.go_to_position(new_pos, new_orien_quaternion)
		
		
	def set_action(self, action):
		assert action.shape == (self.nb_action_values,), 'action shape error'
		
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		cur_orien_euler = p.getEulerFromQuaternion(cur_orien)
		
		new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		
		if self.database.type_db_load == 2:
			# if type is frite parameters
			if self.fix_initial_gripper_orientation:
				new_orien_quaternion = cur_orien
			else:
				new_orien_euler = cur_orien_euler + np.array(action[3:]) * self.max_vel * self.dt
				new_orien_quaternion = p.getQuaternionFromEuler(new_orien_euler)
		else:
			new_orien_euler = cur_orien_euler + np.array(action[3:]) * self.max_vel * self.dt
			new_orien_quaternion = p.getQuaternionFromEuler(new_orien_euler)
			
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, new_orien_quaternion)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=self.joint_motor_control_force * 240.)
	
	def get_obs_ros(self):
		
		self.mutex_observation.acquire()
		try:
			
			tip_position = self.tip_position.copy()
			
			tip_velocity = self.tip_velocity.copy()
			
			tip_orientation = self.tip_orientation.copy()
			
			tip_velocity_orientation = self.tip_velocity_orientation.copy()
			
		finally:
			self.mutex_observation.release()
			
			
		self.mutex_array_mocap_in_arm_frame.acquire()
		try:
			poses_meshes_in_arm_frame = self.poses_meshes_in_arm_frame[1:5,0:3].copy()
			
		finally:
			self.mutex_array_mocap_in_arm_frame.release()	
		
		print("----->", poses_meshes_in_arm_frame.flatten())
		obs = np.concatenate((
					tip_position, tip_orientation, tip_velocity, tip_velocity_orientation, poses_meshes_in_arm_frame.flatten().astype('float64'), self.goal.flatten()
		))
		
		return obs
		
	
	def get_obs(self):
		eff_link_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx, computeLinkVelocity=1)
		gripper_link_pos = np.array(eff_link_state[0]) # gripper cartesian world position = 3 float (x,y,z) = achieved goal
		gripper_link_orien = np.array(eff_link_state[1]) # gripper cartesian world orientation = 4 float (quaternion)
		gripper_link_orien_euler = p.getEulerFromQuaternion(gripper_link_orien) # gripper cartesian world orientation = 3 float (theta_x, theta_y, theta_z)
		gripper_link_vel = np.array(eff_link_state[6]) # gripper cartesian world velocity = 3 float (vx, vy, vz)
		gripper_link_vel_orien = np.array(eff_link_state[7]) # gripper cartesian world angular velocity = 3 float (theta_dot_x, theta_dot_y, theta_dot_z)
		
		self.compute_mesh_pos_to_follow(draw_normal=False)
		mesh_to_follow_pos = np.array(self.position_mesh_to_follow).flatten()
		
		# self.goal = len(self.id_frite_to_follow = [53, 129, 101, 179]) x 3 values (x,y,z) cartesian world position = 12 floats
		# observation = 
		#  3 floats (x,y,z) gripper link cartesian world position  [0,1,2]
		# + 3 float (theta_x,theta_y,theta_z) gripper link cartesian world orientation [3,4,5]
		# + 3 float (vx, vy, vz) gripper link cartesian world velocity [6,7,8]
		# + 3 float (theta__dot_x,theta__dot_y,theta__dot_z) gripper link cartesian world angular velocity [9,10,11]
		# + current cartesian world position of id frite to follow (12 floats) [12,13,14,15,16,17,18,19,20,21,22,23]
		# + self.goal cartesian world position of id frite to reach (12 floats) [24,25,26,27,28,29,30,31,32,33,34,35]
		# observation = 36 floats
		
		#print("goal = {}".format(self.goal))
		#print("goal flat = {}, id pos flat = {}".format(self.goal.flatten(), id_frite_to_follow_pos))
		
		if self.database.type_db_load == 2:
			
			if self.fix_initial_gripper_orientation:
				obs = np.concatenate((gripper_link_pos, gripper_link_vel, mesh_to_follow_pos, self.goal.flatten()))
			else:
				obs = np.concatenate((
							gripper_link_pos, gripper_link_orien_euler, gripper_link_vel, gripper_link_vel_orien, mesh_to_follow_pos, self.goal.flatten())
							)
				
			if self.add_frite_parameters_to_observation:
				# with frite parameters
				obs = np.concatenate(( obs, np.array([float(self.E), float(self.NU)]) ))
			
		else:
			obs = np.concatenate((
						gripper_link_pos, gripper_link_orien_euler, gripper_link_vel, gripper_link_vel_orien, mesh_to_follow_pos, self.goal.flatten()
			))
		
		return obs
	
	def is_success(self, d):
		return (d < self.distance_threshold).astype(np.float32)
		
	def step_ros(self, action):
		action = np.clip(action, self.action_space.low, self.action_space.high)
		new_gripper_pos = self.set_action_ros(action)
		
		time.sleep(self.time_set_action)
		
		obs = self.get_obs_ros()

		done = True
		
		nb_mesh_to_follow = len(self.position_mesh_to_follow)
		
		max_d = 0
		
		initial_pos_of_mesh = 0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(12+(i*3)):(12+(i*3)+3)]
			goal_pos_id_frite = self.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			if (d > max_d):
				max_d = d

		info = {
			'is_success': self.is_success(max_d),
			'max_distance_error' : max_d,
		}
		
		reward = -max_d
		if (max_d > self.distance_threshold):
			done = False

		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info
		
		
	def step(self, action):
		action = np.clip(action, self.action_space.low, self.action_space.high)
		new_gripper_pos = self.set_action(action)
		
		time.sleep(self.time_set_action)
		
		obs = self.get_obs()

		done = True
		
		nb_mesh_to_follow = len(self.position_mesh_to_follow)
		
		max_d = 0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(self.pos_of_mesh_in_obs+(i*3)):(self.pos_of_mesh_in_obs+(i*3)+3)]
			goal_pos_id_frite = self.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			if (d > max_d):
				max_d = d

		info = {
			'is_success': self.is_success(max_d),
			'max_distance_error' : max_d,
		}

		reward = -max_d
		if (max_d > self.distance_threshold):
			done = False

		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info
		
	def reset_env(self, use_frite=True):
		self.debug_gui.reset()
		
		self.env_pybullet.reset()
		
		# load plane
		self.load_plane()
		#p.stepSimulation()

		#load panda
		self.load_panda()
		#p.stepSimulation()
		
		# set panda joints to initial positions
		self.set_panda_initial_joints_positions()
		#p.stepSimulation()
		
		#self.draw_gripper_position()
		#p.stepSimulation()
		
		# load cube
		self.load_cube()
		#p.stepSimulation()
		
		# set gym spaces
		self.set_gym_spaces()
		
		if use_frite:
			# load frite
			self.load_frite()
			#p.stepSimulation()
			
			# anchor frite to gripper
			self.create_anchor_panda()
			#p.stepSimulation()
			
			# close gripper
			#self.close_gripper()
			#p.stepSimulation()
	
	
		if self.database.type_db_load == 2 and self.fix_initial_gripper_orientation:
				self.set_panda_to_initial_gripper_orientation()
			
	
	def reset_ros(self):
		
		
		self.set_gym_spaces_ros()
		
		# sample a new goal
		self.goal = self.sample_goal()
		
		return self.get_obs_ros()
		
		
	def reset(self):
		# sample a new goal
		self.goal = self.sample_goal()
		#p.stepSimulation()
				
		if self.database.type_db_load == 2:
			print("reset env for a db type 2 !")
			self.reset_env()
		
		if self.gui:
			# draw goal
			self.draw_goal()
				
		return self.get_obs()
		
	def render(self):
		print("render !")
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
        

	
