import os, inspect
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

import math
from numpy import linalg as LA

from datetime import datetime



from gym_panda_frite.envs.debug_gui import Debug_Gui

class PandaFriteEnv(gym.Env):
	
	def __init__(self, database = None, distance_threshold = None, gui = None, E = None):
		
		self.database = database
		self.debug_lines_gripper_array = [0, 0, 0, 0]
		self.E = E
		
		# bullet paramters
		#self.timeStep=1./240
		#self.timeStep = 0.003
		self.timeStep = 0.0001
		self.n_substeps = 20
		self.dt = self.timeStep*self.n_substeps
		self.max_vel = 1
		self.max_gripper_vel = 20
		
		
		self.id_debug_gripper_position = None
		self.id_debug_joints_values = None
		self.id_debug_frite_list = None
		self.id_debug_marker_frite_list = None
		self.id_save_button = None
		
		
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
		self.id_frite_to_follow = [ [31, 15], [13, 10], [18, 14], [28, 53] ]  # left then right  [left, right], [left,right] ...
		#self.id_frite_to_follow = [ [31, 15], [47, 33], [18, 14], [28, 53] ]  # left then right  [left, right], [left,right] ...
		
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
		self.under_id_frite_to_follow = [ [63, 38], [58, 54], [42, 37], [23, 32] ]  # left then right  [left, right], [left,right] ...
		#self.under_id_frite_to_follow = [ [63, 38], [64, 45], [42, 37], [23, 32] ]  # left then right  [left, right], [left,right] ...
		
		# array containing the upper mean point shifted by a normalized normal vector
		self.position_mesh_to_follow = [None, None, None, None]
		
		# array containing the upper mean points (between left and right upper points)
		self.mean_position_to_follow = [None, None, None, None]
		
		#self.debug_id_frite_to_follow = [[None,None],[None,None]]  # draw 2 lines (a cross) per id frite to follow
		
		self.debug_gui = Debug_Gui(env = self)
		
		self.distance_threshold=distance_threshold
		
		#print("PandaFriteEnv distance_threshold = {}".format(self.distance_threshold))
		
		self.seed()
		
		if gui == True:
			# connect bullet
			p.connect(p.GUI) #or p.GUI (for test) or p.DIRECT (for train) for non-graphical version
		else:
			p.connect(p.DIRECT)
		
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
		
		self.reset(use_frite=True)
		
		self.panda_list_lower_limits, self.panda_list_upper_limits, self.panda_list_joint_ranges, self.panda_list_initial_poses = self.get_panda_joint_ranges()
			 
		# show sliders
		self.panda_joint_name_to_slider={}
		#self.show_sliders()
		
		#self.show_cartesian_sliders()
		
		p.stepSimulation()
	
	
	
	def to_rt_matrix(self, Q, T):
		# Extract the values from Q
		q0 = Q[0]
		q1 = Q[1]
		q2 = Q[2]
		q3 = Q[3]
		 
		# First row of the rotation matrix
		r00 = 2 * (q0 * q0 + q1 * q1) - 1
		r01 = 2 * (q1 * q2 - q0 * q3)
		r02 = 2 * (q1 * q3 + q0 * q2)
		 
		# Second row of the rotation matrix
		r10 = 2 * (q1 * q2 + q0 * q3)
		r11 = 2 * (q0 * q0 + q2 * q2) - 1
		r12 = 2 * (q2 * q3 - q0 * q1)
		 
		# Third row of the rotation matrix
		r20 = 2 * (q1 * q3 - q0 * q2)
		r21 = 2 * (q2 * q3 + q0 * q1)
		r22 = 2 * (q0 * q0 + q3 * q3) - 1
		 
		# 4x4 RT matrix
		rt_matrix = np.array([[r00, r01, r02, T[0]],
							   [r10, r11, r12, T[1]],
							   [r20, r21, r22, T[2]],
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
				print("id={}, normal={}".format(self.id_frite_to_follow[i][0],a_normal_pt))
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
			self.debug_gui.draw_cross("goal_"+str(i) , a_pos = self.goal[i])
			
			
	def draw_gripper_position(self):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		self.debug_gui.draw_cross("gripper", a_pos = cur_pos, a_color = [0, 0, 1])
	
	def sample_goal_random(self):
		# sample a goal np.array[x,v,z] from the goal_space 
		goal = np.array(self.goal_space.sample())
		return goal.copy()
	
	def sample_goal(self):
		return self.database.get_random_targets()
	
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
		p.stepSimulation()

	
	def set_gym_spaces(self):
		panda_eff_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		"""
		# MEDIUM
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
		# SMALL
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-3*low_marge
		low_y_up = panda_eff_state[0][1]+3*low_marge
		
		
		z_low_marge = 0.25
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		
		
		# EXTRA EXTRA SMALL
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
		# EXTRA SMALL
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.0*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-2.5*low_marge
		low_y_up = panda_eff_state[0][1]+2.5*low_marge
		
		
		#z_low_marge = 0.25
		z_low_marge = 0.10
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		self.goal_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		#print("frite env goal space = {}".format(self.goal_space))
		
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-2*low_marge
		low_x_up = panda_eff_state[0][0]+low_marge

		low_y_down = panda_eff_state[0][1]-5*low_marge
		low_y_up = panda_eff_state[0][1]+5*low_marge

		z_low_marge = 0.3
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		
		self.pos_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		
		# action_space = cartesian world velocity (vx, vy, vz)  = 3 float
		self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
		
		# observation = 32 float -> see function _get_obs
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(30,), dtype=np.float32)

		
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
	
	
	def printCurrentEndEffPosition(self):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		print("End Eff position => x={:.3f}, y={:.3f}, z={:.3f} ".format(cur_pos[0], cur_pos[1], cur_pos[2]))
		print("End Eff orientation => x={:.3f}, y={:.3f}, z={:.3f}, w={:.3f} ".format(cur_orien[0], cur_orien[1], cur_orien[2], cur_orien[3]))
		
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
		
		# plage E -> 0.1 ?? 40
		# frite blanche :
		# E = 0.1*pow(10,6)  NU = 0.49
		
		# frite noire :
		# E = 40*pow(10,6)  NU = 0.49
		
		E = self.E*pow(10,6)
		NU = 0.49
		(a_lambda,a_mu) = self.conv_module_d_young_to_lame(E,NU)
		
		# frite : 103 cm with 0.1 cell size
		self.frite_id = p.loadSoftBody("vtk/frite.vtk", basePosition = self.frite_startPos, baseOrientation=self.frite_startOrientation, mass = 0.2, useNeoHookean = 1, NeoHookeanMu = a_mu, NeoHookeanLambda = a_lambda, NeoHookeanDamping = 0.01, useSelfCollision = 1, collisionMargin = 0.001, frictionCoeff = 0.5, scale=1.0)
		#p.changeVisualShape(self.frite_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
			
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
				p.stepSimulation()


	def close_gripper(self):
		id_finger_joint1  = self.panda_joint_name_to_ids['panda_finger_joint1']
		id_finger_joint2  = self.panda_joint_name_to_ids['panda_finger_joint2']

		p.setJointMotorControl2(self.panda_id, id_finger_joint1, 
								p.POSITION_CONTROL,targetPosition=0.025,
								positionGain=0.1)
								
		p.setJointMotorControl2(self.panda_id, id_finger_joint2 , 
								p.POSITION_CONTROL,targetPosition=0.025,
								positionGain=0.1)
								
    
	def create_anchor_panda(self):
		
		# mesh id = 0 -> down
		# -1, -1 -> means anchor to the cube
		p.createSoftBodyAnchor(self.frite_id, 0, self.cube_id , -1, [0,0,0])
		#p.createSoftBodyAnchor(self.frite_id, 0, -1 , -1)
		
		# mesh id = 1 -> up
		p.createSoftBodyAnchor(self.frite_id, 1, self.panda_id , self.panda_end_eff_idx, [0,0,0])
		
		# panda finger joint 1 = 10
		# panda finger joint 2 = 11
		
		"""
		pos_10 = p.getLinkState(self.panda_id, 10)[0]
		new_pos_10  = [pos_10[0]+0.025, pos_10[1]+0.056, pos_10[2]-0.02]
		pos_11 = p.getLinkState(self.panda_id, 11)[0]
		new_pos_11  = [pos_11[0]+0.025, pos_11[1]-0.056, pos_11[2]-0.02]
		self.debug_gui.draw_cross("pos_10" , a_pos = new_pos_10, a_size = 0.02)
		self.debug_gui.draw_cross("pos_11" , a_pos = new_pos_11, a_size = 0.02)
		p.stepSimulation()
		
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		pos_frite_6 = data[1][6]
		pos_frite_9 = data[1][9]
		self.debug_gui.draw_cross("pos_frite_6" , a_pos = pos_frite_6, a_size = 0.02)
		self.debug_gui.draw_cross("pos_frite_9" , a_pos = pos_frite_9, a_size = 0.02)
		"""
		
		
		#p.createSoftBodyAnchor(self.frite_id, 6, self.panda_id , 10, [0.025,0.056,-0.02])
		#p.createSoftBodyAnchor(self.frite_id, 9, self.panda_id , 11, [0.025,-0.056,-0.02])
		
		
		p.createSoftBodyAnchor(self.frite_id, 6, self.panda_id , 10, [0.025,-0.01,-0.02])
		p.createSoftBodyAnchor(self.frite_id, 9, self.panda_id , 11, [0.025,0.01,-0.02])
	
	def go_to_position_simulated(self, a_position):
		
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = a_position
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=100 * 240.)
			p.stepSimulation()
		
			
		for i in range(1000):
					p.stepSimulation()
		
		"""	
		start=datetime.now()
		
		while True:
			p.stepSimulation()
			cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
			cur_pos = np.array(cur_state[0])
		
			if np.linalg.norm(cur_pos - new_pos, axis=-1) <= 0.03:
				print("Exit with precision !")
				break
			
			if (datetime.now()-start).total_seconds() >= 60:
				print("Exit with time !")
				break
		"""
	def set_action_cartesian(self, action):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = cur_pos + np.array(action)
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=100 * 240.)
			
		
	def set_action(self, action):
		assert action.shape == (3,), 'action shape error'
		
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=100 * 240.)
		
		"""
		for i in range(1000):
					p.stepSimulation()
		"""
	
	def get_obs(self):
		eff_link_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx, computeLinkVelocity=1)
		gripper_link_pos = np.array(eff_link_state[0]) # gripper cartesian world position = 3 float (x,y,z) = achieved goal
		gripper_link_vel = np.array(eff_link_state[6]) # gripper cartesian world velocity = 3 float (vx, vy, vz)
		
		mesh_to_follow_pos = np.array(self.position_mesh_to_follow).flatten()
		
		# self.goal = len(self.id_frite_to_follow = [53, 129, 101, 179]) x 3 values (x,y,z) cartesian world position = 12 floats
		# observation = 
		#  3 floats (x,y,z) gripper link cartesian world position  [0,1,2]
		# + 3 float (vx, vy, vz) gripper link cartesian world velocity [3,4,5]
		# + current cartesian world position of id frite to follow (12 floats) [6,7,8,9,10,11,12,13,14,15,16,17]
		# + self.goal cartesian world position of id frite to reach (12 floats) [18,19,20,21,22,23,24,25,26,27,28,29]
		# observation = 30 floats
		
		#print("goal = {}".format(self.goal))
		#print("goal flat = {}, id pos flat = {}".format(self.goal.flatten(), id_frite_to_follow_pos))
		
		obs = np.concatenate((
					gripper_link_pos, gripper_link_vel, mesh_to_follow_pos, self.goal.flatten()
		))
		
		return obs
	
	def is_success(self, d):
		return (d < self.distance_threshold).astype(np.float32)
		
	def step(self, action):
		action = np.clip(action, self.action_space.low, self.action_space.high)
		new_gripper_pos = self.set_action(action)
		
		p.stepSimulation()
		
		obs = self.get_obs()

		done = True
		
		nb_mesh_to_follow = len(self.position_mesh_to_follow)
		
		sum_d = 0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(6+(i*3)):(6+(i*3)+3)]
			goal_pos_id_frite = self.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			sum_d+=d
			
		
		d = np.float32(sum_d/nb_mesh_to_follow)

		info = {
			'is_success': self.is_success(d),
			'mean_distance_error' : d,
		}

		reward = -d
		if (d > self.distance_threshold):
			done = False

		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info
		
	def reset(self, use_frite=True):	
		
		# reset pybullet to deformable object
		p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
		
		# bullet setup
		# add pybullet path
		currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		#print("currentdir = {}".format(currentdir))
		p.setAdditionalSearchPath(currentdir)
		
		p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
		p.setTimeStep(self.timeStep)

		# Set Gravity to the environment
		#p.setGravity(0, 0, -9.81)
		p.setGravity(0, 0, 0)
		
		# load plane
		self.load_plane()
		p.stepSimulation()

		
		#load panda
		self.load_panda()
		p.stepSimulation()
		
		# set panda joints to initial positions
		self.set_panda_initial_joints_positions()
		p.stepSimulation()
		
		
		self.draw_gripper_position()
		p.stepSimulation()
		
		# load cube
		self.load_cube()
		p.stepSimulation()
		
		# set gym spaces
		self.set_gym_spaces()
		
		if use_frite:
			# load frite
			self.load_frite()
			p.stepSimulation()
		
			
			# close gripper
			self.close_gripper()
			p.stepSimulation()
			
			
			# anchor frite to gripper
			self.create_anchor_panda()
			p.stepSimulation()
		
		
		# sample a new goal
		#self.goal = self.sample_goal()
		#p.stepSimulation()
		
		#self.draw_all_ids_mesh_frite()
		#p.stepSimulation()
		
		#self.compute_mean_pos_id_frite_to_follow()
		#p.stepSimulation()
		
		#self.compute_mesh_pos_to_follow()
		#p.stepSimulation()
		
		# draw goal
		#self.draw_goal()
		#p.stepSimulation()
		
		#return self.get_obs()
		
	def render(self):
		print("render !")
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
        

	
