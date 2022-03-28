import time

from numpy import linalg as LA

import numpy as np


def to_rt_matrix(x, y, z, w):
	
	# to test : https://www.andre-gaschler.com/rotationconverter/
	# Extract the values from Q
	qw = w
	qx = x
	qy = y
	qz = z
	
	# https://pybullet.org/Bullet/BulletFull/btMatrix3x3_8h_source.html
	
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
	
	
	print('\n',r00,' , ', r01, ' , ', r02 , '\n', r10, ' , ', r11, ' , ', r12, '\n', r20, ' , ', r21, ' , ', r22) 
	

if __name__ == '__main__':
	print('----------------------------------')
	to_rt_matrix(0.0, 0.0, 0.0, 1.0)
	print('----------------------------------')
	to_rt_matrix(0.23912, 0.36964, 0.099046, 0.8924)
	print('----------------------------------')
	to_rt_matrix(0.23912, 0.36964, 0.099046, 1.0)
	print('----------------------------------')
	to_rt_matrix(0.55, 0.2, 0.0, 0.8924)
	print('----------------------------------')
