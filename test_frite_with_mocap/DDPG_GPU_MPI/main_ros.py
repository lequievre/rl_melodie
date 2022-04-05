				
import sys
import gym
import numpy as np
import random
import torch
import os

import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

import argparse
import os
import time
from datetime import datetime

from mpi4py import MPI
import pybullet as p

import gym_panda_frite
from database_frite import Database_Frite

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test' or 'debug_cartesian' or 'debug_articular'
parser.add_argument("--env_name", default="PandaFrite-v1")
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--max_episode', default=2000, type=int) # num of episodes
parser.add_argument('--max_step', default=500, type=int) # num of step per episodes
parser.add_argument('--batch_size', default=128, type=int) # num of games
parser.add_argument('--max_memory_size', default=50000, type=int) # num of games
parser.add_argument('--update_iteration', default=10, type=int) # num of games
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--save_dir_name', default='./weights/', type=str)
parser.add_argument('--cuda', default=False, type=bool) # use cuda
parser.add_argument('--generate_database_name', default='database_id_frite.txt', type=str)
parser.add_argument('--load_database_name', default='database_id_frite.txt', type=str)
parser.add_argument('--distance_threshold', default=0.05, type=float) #
parser.add_argument('--generate_db_dir_name', default='/default_generate/', type=str)
parser.add_argument('--load_db_dir_name', default='/default_load/', type=str)
parser.add_argument('--db_nb_x', default=8, type=int)
parser.add_argument('--db_nb_y', default=22, type=int)
parser.add_argument('--db_nb_z', default=10, type=int)
parser.add_argument('--gui', default=False, type=bool) # use cuda

args = parser.parse_args()

def main():

	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'

	directory = args.save_dir_name
	root_path_databases = "./databases"
	generate_path_databases = root_path_databases + args.generate_db_dir_name
	load_path_databases = root_path_databases + args.load_db_dir_name

	rank = MPI.COMM_WORLD.Get_rank()

	if MPI.COMM_WORLD.Get_rank() == 0:
		if not os.path.isdir(directory):
			os.makedirs(directory)
		if not os.path.isdir(generate_path_databases):
			os.makedirs(generate_path_databases)
		if not os.path.isdir(load_path_databases):
			os.makedirs(load_path_databases)

	if not os.path.isfile(load_path_databases + args.load_database_name):
		   raise RuntimeError("=> Database file to load does not exit : " + load_path_databases + args.load_database_name)
		   return        

	db = Database_Frite(path_load=load_path_databases, load_name=args.load_database_name, generate_name=args.generate_database_name, path_generate=generate_path_databases, nb_x=args.db_nb_x, nb_y=args.db_nb_y, nb_z=args.db_nb_z)
	env = gym.make(args.env_name, database=db, distance_threshold=args.distance_threshold, gui=args.gui)

	env.seed(args.random_seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(args.random_seed + MPI.COMM_WORLD.Get_rank())
	np.random.seed(args.random_seed + MPI.COMM_WORLD.Get_rank())

	if (args.cuda):
		torch.cuda.manual_seed(args.random_seed + MPI.COMM_WORLD.Get_rank())

	agent = DDPGagent(args.cuda, env, max_memory_size=args.max_memory_size, directory=directory)
	noise = OUNoise(env.action_space)
    
	env.init_ros()
	#env.generate_mocap_databases()
	env.load_database_mocap()
	
	input("Press Enter to stop !")

	#state = env.reset(use_frite=True)
	"""
	while True:
		keys = p.getKeyboardEvents()
		if 65309 in keys:
		   break	
	"""
        
if __name__ == '__main__':
    main()

