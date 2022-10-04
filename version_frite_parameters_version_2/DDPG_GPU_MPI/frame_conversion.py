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
	
	# to test : https://www.andre-gaschler.com/rotationconverter/
	
	# Extract the values from Q
	qw = Q.w
	qx = Q.x
	qy = Q.y
	qz = Q.z
	
	# https://pybullet.org/Bullet/BulletFull/btMatrix3x3_8h_source.html
	# https://www.tabnine.com/web/assistant/code/rs/5c7cb2fc2ef5570001df5f0e#L546
	# https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
	# https://lucidar.me/fr/quaternions/quaternion-normalization/
	
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
	
	
	""" 
	# First row of the rotation matrix
	# https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
	# [1 - 2*qy^2 - 2*qz^2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw]
	# [2*qx*qy + 2*qz*qw, 1 - 2*qx^2 - 2*qz^2, 2*qy*qz - 2*qx*qw]
	# [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx^2 - 2*qy^2]
	
	

	r00 = 1 - 2 * (qy * qy + qz * qz)
	r01 = 2 * (qx * qy - qz * qw)
	r02 = 2 * (qx * qz + qy * qw)
	 
	# Second row of the rotation matrix
	r10 = 2 * (qx * qy + qz * qw)
	r11 = 1 - 2 * (qx * qx + qz * qz)
	r12 = 2 * (qy * qz - qx * qw)
	 
	# Third row of the rotation matrix
	r20 = 2 * (qx * qz - qy * qw)
	r21 = 2 * (qy * qz + qx * qw)
	r22 = 1- 2 * (qx * qx + qy * qy)
	 
	# 4x4 RT matrix
	rt_matrix = np.array([[r00, r01, r02, T.x],
						   [r10, r11, r12, T.y],
						   [r20, r21, r22, T.z],
						   [0, 0, 0, 1]])
	"""
							
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
    
    
