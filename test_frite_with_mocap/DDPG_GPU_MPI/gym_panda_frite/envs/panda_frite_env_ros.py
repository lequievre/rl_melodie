import os, inspect
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

import math
from numpy import linalg as LA

import rospy
import rospkg
import threading

from geometry_msgs.msg import PoseArray, Point, Quaternion
from geometry_msgs.msg import PoseStamped

from visualization_msgs.msg import MarkerArray, Marker

from gym_panda_frite.envs.debug_gui import Debug_Gui

class PandaFriteEnvROS(gym.Env):
	
	def __init__(self, database = None, distance_threshold = None, gui = None):
		
		print("****** ROSSSSSSSSSS !!!! ************")
		
		self.database = database
		self.debug_lines_gripper_array = [0, 0, 0, 0]
		
		# bullet paramters
		#self.timeStep=1./240
		self.timeStep = 0.003
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
		# [31, 15]
		#   ||
		#   vv
		# [13, 10]
		#   ||
		#   vv
		# [18, 14]
		#   ||
		#   vv
		# [9, 6] (TIP)
		self.id_frite_to_follow = [ [31, 15], [13, 10], [18, 14], [9, 6] ]  # left then right  [left, right], [left,right] ...
		
		# Points from bottom to up, on the same plane of id_frite_to _follow, one level under ((on the front side)
		# [63, 38] (under)
		# [31, 15]
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
		# [28, 53] (under)
		# [9, 6] (TIP)
		self.under_id_frite_to_follow = [ [63, 38], [58, 54], [42, 37], [28, 53] ]  # left then right  [left, right], [left,right] ...
		
		# mean tip between 28,53 (on front side) and 27, 26 (on back side)
		# 28 is front, 27 is back
		# 53 is front, 26 is back
		#     27------26
		#   .        .
		#  .        .
		# 28------53
		self.id_frite_locate_tip = [ [28, 27], [53, 26] ] # [left,left] then [right,right]
		
		# array containing the upper mean point shifted by a normalized normal vector
		self.position_mesh_to_follow = [None, None, None, None]
		
		# array containing the upper mean points (between left and right upper points)
		self.mean_position_to_follow = [None, None, None, None]
		
		# Numpy array that content the Poses and orientation of each rigid body defined with the mocap
		# [[mesh0 geometry_msgs/Pose ], [mesh1 geometry_msgs/Pose] ..., [mesh n geometry_msgs/Pose]]
		self.array_mocap_poses_base_frame = None
		
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

	def draw_cross_mocap_mesh(self):
		for i in range(len(self.poses_meshes_in_arm_frame)):
			self.debug_gui.draw_cross("mesh_mocap_" + str(i) , a_pos = [self.poses_meshes_in_arm_frame[i][0],self.poses_meshes_in_arm_frame[i][1],self.poses_meshes_in_arm_frame[i][2]]
		)
	
	def open_database_mocap(self):
		self.file_goal_mocap_poses = open("database_goal_mocap_poses.txt", "w+")
		
	def close_database_mocap(self):
		self.file_goal_mocap_poses.close()
		
	def generate_mocap_databases(self):
		copy_of_array_mocap_poses_base_frame = None
		nb_goal_to_sample = 50
		self.open_database_mocap()
		
		for i in range(nb_goal_to_sample):
			a_goal = self.sample_goal_database()
			self.publish_position(a_goal)
			
			# wait a time to reach the 'goal' position
			time.sleep(4)
			
			self.mutex_array_mocap.acquire()
			try:
				copy_of_array_mocap_poses_base_frame = self.array_mocap_poses_base_frame.copy()
			finally:
				self.mutex_array_mocap.release()
				
			self.file_goal_mocap_poses.write("{:.3f} {:.3f} {:.3f}".format(a_goal[0], a_goal[1], a_goal[2]))
			for a_pose in copy_of_array_mocap_poses_base_frame:
				self.file_goal_mocap_poses.write(" {:.3f} {:.3f} {:.3f}".format(a_pose.position.x, a_pose.position.y, a_pose.position.z))
			self.file_goal_mocap_poses.write("\n")
			
		self.close_database_mocap()
	
	def sample_goal_database(self):
		# sample a goal np.array[x,v,z] from the goal_space 
		goal = np.array(self.goal_space.sample())
		return goal.copy()
		
		
	def publish_position(self, command):
		pose_msg = PoseStamped()
		pose_msg.pose.position.x = command[0]
		pose_msg.pose.position.y = command[1]
		pose_msg.pose.position.z = command[2]
	
		pose_msg.pose.orientation.x = 0
		pose_msg.pose.orientation.y = 0
		pose_msg.pose.orientation.z = 0
		pose_msg.pose.orientation.w = 1
		
		self.publisher_position.publish(pose_msg)
	
	
	def publish_mocap_mesh(self):
		
		simple_marker_msg = Marker()
		marker_array_msg = MarkerArray()
		marker_array_msg.markers = []
		
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
			simple_marker_msg.color.r = 1.0
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
			
			if i > 0:
				point_msg = Point()
				point_msg.x = self.poses_meshes_in_arm_frame[i][0]
				point_msg.y = self.poses_meshes_in_arm_frame[i][1]
				point_msg.z = self.poses_meshes_in_arm_frame[i][2]
				
				line_strip_msg.points.append(point_msg)
			
		
		self.publisher_poses_meshes_in_arm_frame.publish(marker_array_msg)
		self.publisher_line_strip_in_arm_frame.publish(line_strip_msg)
		
		
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

		for i in range(5):
			pos_mesh_in_mocap_frame = np.array([msg.poses[i].position.x,msg.poses[i].position.y,msg.poses[i].position.z,1])
			self.poses_meshes_in_arm_frame[i] = np.dot(self.matrix_mocap_frame_in_arm_frame, pos_mesh_in_mocap_frame)
				
		#self.draw_cross_mocap_mesh()
		#print(len(self.poses_meshes_in_arm_frame))
		self.publish_mocap_mesh()
		
	def init_ros(self):
		rospy.init_node('rl_melodie_node')
		
		self.matrix_base_frame_in_mocap_frame = None
		self.matrix_mocap_frame_in_arm_frame = None
										
										
		self.matrix_base_frame_in_arm_frame = np.array(
											[[1, 0, 0, 0.025],
											[0, 1, 0, 0.422],
											[0, 0, 1, 0.017],
											[0, 0, 0, 1]]
										)
										
		self.poses_meshes_in_arm_frame = np.array([[None, None, None, None],[None, None, None, None],[None, None, None, None],[None, None, None, None],[None, None, None, None]])
		
		rospy.Subscriber('/PoseAllBodies', PoseArray, self.mocap_callback,
						 queue_size=10)
						 
		self.publisher_poses_meshes_in_arm_frame = rospy.Publisher('/VisualizationPoseArrayMarkersInArmFrame', MarkerArray, queue_size=10)
		
		self.publisher_line_strip_in_arm_frame = rospy.Publisher('/VisualizationLineStripMarkerInArmFrame', Marker, queue_size=10)
		
		self.publisher_position = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
		
		self.mutex_array_mocap = threading.Lock()
	
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
		for i in range(len(self.id_frite_to_follow)-1):
			
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
			
		# Compute only for the TIP
		# The real TIP  position is inside the "frite"
		# mean tip between 28, 53 (front) and 27, 26 (back)
		# 28 is front, 27 is back (left)
		# 53 is front, 26 is back (right)
		# mean between 28,53 and 27, 26
		pt_left_1 = np.array(data[1][self.id_frite_locate_tip[0][0]])  # 28
		pt_left_2 = np.array(data[1][self.id_frite_locate_tip[0][1]])  # 27
		pt_mean_left = (pt_left_1 + pt_left_2) / 2.0  # point left inside 'frite'
		
		pt_right_1 = np.array(data[1][self.id_frite_locate_tip[1][0]]) # 53
		pt_right_2 = np.array(data[1][self.id_frite_locate_tip[1][1]]) # 26
		pt_mean_right = (pt_right_1 + pt_right_2) / 2.0 # point right inside 'frite'
		
		# Get the real position of the TIP from pybullet (inside the 'frite')
		self.mean_position_to_follow[len(self.position_mesh_to_follow)-1] = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		
		# 0.025 is equal to the half of the "frite" width
		# 0.007 is equal to the half of the marker thickness
		self.position_mesh_to_follow[len(self.position_mesh_to_follow)-1] = self.shift_point_in_normal_direction(pt_left=pt_mean_left, pt_right=pt_mean_right, pt_mean=self.mean_position_to_follow[len(self.position_mesh_to_follow)-1], a_distance = (0.007 + 0.025))
		
		if draw_normal:
			a_normal_pt = self.shift_point_in_normal_direction(pt_left=pt_mean_left, pt_right=pt_mean_right, pt_mean=self.mean_position_to_follow[len(self.position_mesh_to_follow)-1], a_distance = 0.1)
			self.draw_normal_plane(len(self.position_mesh_to_follow)-1, data, a_normal_pt)
				
	def draw_cross_mesh_to_follow(self):
		for i in range(len(self.position_mesh_to_follow)):
			self.debug_gui.draw_cross("mesh_frite_" + str(i) , a_pos = self.position_mesh_to_follow[i])
			
	def compute_height_id_frite(self):
		#self.id_frite_to_follow = [15, 10, 14]
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		pos_2 = data[1][2]
		pos_15 = data[1][15]
		pos_10 = data[1][10]
		pos_14 = data[1][14]
		
		pos_1 = data[1][1]
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		
		z_diff_1_gripper = pos_1[2] - gripper_pos[2]
		
		
		z_diff_2_15 = pos_2[2] - pos_15[2]
		z_diff_15_10 = pos_15[2] - pos_10[2]
		z_diff_10_14 = pos_10[2] - pos_14[2]
		
		self.debug_gui.draw_text("z_diff_2_15", a_text = "z_diff_2_15=" + str(z_diff_2_15), a_pos=[1,1,1])
		self.debug_gui.draw_text("z_diff_15_10", a_text = "z_diff_15_10=" + str(z_diff_15_10), a_pos=[1,1,1.5])
		self.debug_gui.draw_text("z_diff_10_14", a_text = "z_diff_10_14=" + str(z_diff_10_14), a_pos=[1,1,2.0])
		
		self.debug_gui.draw_text("z_diff_1_gripper", a_text = "z_diff_1_gripper=" + str(z_diff_1_gripper), a_pos=[1,1,2.5])
		
		
		
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
		
		# frite : 103 cm with 0.1 cell size
		self.frite_id = p.loadSoftBody("vtk/frite.vtk", basePosition = self.frite_startPos, baseOrientation=self.frite_startOrientation, mass = 0.2, useNeoHookean = 1, NeoHookeanMu = 961500, NeoHookeanLambda = 1442300, NeoHookeanDamping = 0.01, useSelfCollision = 1, collisionMargin = 0.001, frictionCoeff = 0.5, scale=1.0)
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
		"""pos_10 = p.getLinkState(self.panda_id, 10)[0]
		new_pos_10  = [pos_10[0]+0.025, pos_10[1]-0.01, pos_10[2]-0.02]
		pos_11 = p.getLinkState(self.panda_id, 11)[0]
		new_pos_11  = [pos_11[0]+0.025, pos_11[1]+0.01, pos_11[2]-0.02]"""
		p.createSoftBodyAnchor(self.frite_id, 6, self.panda_id , 10, [0.025,-0.01,-0.02])
		p.createSoftBodyAnchor(self.frite_id, 9, self.panda_id , 11, [0.025,0.01,-0.02])
		
		
	def set_action_cartesian(self, action):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = cur_pos + np.array(action)
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=10 * 240.)
			
		
	def set_action(self, action):
		assert action.shape == (3,), 'action shape error'
		
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=10 * 240.)
	
	
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
        

	
