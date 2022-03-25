import rospy

from qualisys.msg import Subject
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

import time

from numpy import linalg as LA

import numpy as np

matrix_base_frame_in_arm_frame = np.array(
											[[1, 0, 0, 0.025],
											[0, 1, 0, 0.422],
											[0, 0, 1, 0.017],
											[0, 0, 0, 1]]
										)
										
matrix_base_frame_in_mocap_frame = 0
matrix_mocap_frame_in_arm_frame = 0

def to_rt_matrix(Q, T):
	# Extract the values from Q
	q0 = Q.w
	q1 = Q.x
	q2 = Q.y
	q3 = Q.z
	 
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
	rt_matrix = np.array([[r00, r01, r02, T.x],
						   [r10, r11, r12, T.y],
						   [r20, r21, r22, T.z],
						   [0, 0, 0, 1]])
							
	return rt_matrix

def base_frame_callback(msg):
	pos_base_frame = msg.position
	orien_base_frame = msg.orientation
	
	global matrix_base_frame_in_mocap_frame
	matrix_base_frame_in_mocap_frame = to_rt_matrix(orien_base_frame, pos_base_frame)
	
	global matrix_mocap_frame_in_arm_frame
	matrix_mocap_frame_in_arm_frame = np.dot(matrix_base_frame_in_arm_frame, LA.inv(matrix_base_frame_in_mocap_frame))
	
	#print("base frame -> pos =", pos_base_frame, ", orien =", orien_base_frame)
	
def mesh1_callback(msg):
	pos_mesh1 = msg.position
	orien_mesh1 = msg.orientation
	
	
	pos_mesh1_in_mocap_frame = np.array([pos_mesh1.x,pos_mesh1.y,pos_mesh1.z,1])
	
	pos_mesh1_in_arm_frame = np.dot(matrix_mocap_frame_in_arm_frame, pos_mesh1_in_mocap_frame)
	print("mesh1 in arm frame -> ", pos_mesh1_in_arm_frame)
	

def listener():
	rospy.Subscriber('/qualisys_frite/base_frame', Subject, base_frame_callback, queue_size=10)
	rospy.Subscriber('/qualisys_frite/mesh1', Subject, mesh1_callback, queue_size=10)

if __name__ == '__main__':
	# Initialize the node and name it.
	rospy.init_node('frame_conversion_node')
	listener()
		
	rospy.spin()
    
    
