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

from gym_panda_frite.envs.environment import Environment
from gym_panda_frite.envs.wrapper_panda_frite_env import WrapperPandaFriteEnv

from json_decoder import JsonDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--gui', default=False, type=bool) # use gui to see graphic results
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)

args = parser.parse_args()

def main():
	
	json_decoder = JsonDecoder(args.config_file)

	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'

	root_path_databases = json_decoder.config_data["database"]["root_path_databases"]
	generate_db_dir_name = json_decoder.config_data["database"]["generate"]["dir_name"]
	load_db_dir_name = json_decoder.config_data["database"]["load"]["dir_name"]
	load_database_name = json_decoder.config_data["database"]["load"]["database_name"]
	generate_database_name = json_decoder.config_data["database"]["generate"]["database_name"]
	
	generate_path_databases = root_path_databases + generate_db_dir_name
	load_path_databases = root_path_databases + load_db_dir_name
	print("** DATABASE PARAMETERS **")
	print("root_path_databases = {}".format(root_path_databases))
	print("generate_db_dir_name = {}".format(generate_db_dir_name))
	print("load_db_dir_name = {}".format(load_db_dir_name))
	print("load_database_name = {}".format(load_database_name))
	print("generate_database_name = {}".format(generate_database_name))
	print("generate_path_databases = {}".format(generate_path_databases))
	print("load_path_databases = {}".format(load_path_databases))
	
	env_name = json_decoder.config_data["env"]["name"]
	env_random_seed = json_decoder.config_data["env"]["random_seed"]
	env_time_set_action = json_decoder.config_data["env"]["time_set_action"]
	
	print("** ENV PARAMETERS **")
	print("env_name = {}".format(env_name))
	print("env_random_seed = {}".format(env_random_seed))
	print("env_time_set_action = {}".format(env_time_set_action))
	
	weights_dir_name = json_decoder.config_data["ddpg"]["weights_dir_name"]
	ddpg_cuda = json_decoder.config_data["ddpg"]["cuda"]
	ddpg_max_memory_size = json_decoder.config_data["ddpg"]["max_memory_size"]
	ddpg_batch_size = json_decoder.config_data["ddpg"]["batch_size"]
	ddpg_log_interval = json_decoder.config_data["ddpg"]["log_interval"]
	
	print("** DDPG PARAMETERS **")
	print("ddpg_cuda = {}".format(ddpg_cuda))
	print("ddpg_batch_size = {}".format(ddpg_batch_size))
	print("ddpg_max_memory_size = {}".format(ddpg_max_memory_size))
	print("ddpg_log_interval = {}".format(ddpg_log_interval))
	print("weights_dir_name = {}".format(weights_dir_name))
	
	log_path = json_decoder.config_data["log"]["log_path"]
	log_name = json_decoder.config_data["log"]["log_name"]
	
	print("** LOG PARAMETERS **")
	print("log_path = {}".format(log_path))
	print("log_name = {}".format(log_name))
	
	rank = MPI.COMM_WORLD.Get_rank()
    
	if MPI.COMM_WORLD.Get_rank() == 0:
		if not os.path.isdir(weights_dir_name):
			os.makedirs(weights_dir_name)
		if not os.path.isdir(generate_path_databases):
			os.makedirs(generate_path_databases)
		if not os.path.isdir(load_path_databases):
			os.makedirs(load_path_databases)
		if not os.path.isdir(log_path):
			os.makedirs(log_path)
    
	if not os.path.isfile(load_path_databases + load_database_name):
		raise RuntimeError("=> Database file to load does not exit : " + load_path_databases + load_database_name)
		return
		
	if rank == 0:
		file_log = open(log_path + log_name, "w+")
		
	env_pybullet = Environment(json_decoder=json_decoder, gui=args.gui)
	env_pybullet.reset()
	db = Database_Frite(json_decoder=json_decoder)
	env = WrapperPandaFriteEnv(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui))

	env.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	np.random.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	if (ddpg_cuda):
		torch.cuda.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	agent = DDPGagent(ddpg_cuda, env, max_memory_size=ddpg_max_memory_size, directory=weights_dir_name)
	noise = OUNoise(env.action_space)
    
	list_global_rewards = []
    
	start=datetime.now()
	
	n_episodes = json_decoder.config_data["env_train"]["n_episodes"]
	n_steps = json_decoder.config_data["env_train"]["n_steps"]
	do_reset_env = json_decoder.config_data["env"]["do_reset_env"]
	ddpg_load = json_decoder.config_data["ddpg"]["load"]
	
	print("** ENV MODE TRAIN **")
	print("n_episodes = {}".format(n_episodes))
	print("n_steps = {}".format(n_steps))
	print("do_reset_env = {}".format(do_reset_env))
	
	if rank == 0:
		file_log.write("** ENV MODE TRAIN **\n")
		file_log.write("config_file = {}\n".format(args.config_file))
		file_log.write("n_episodes = {}\n".format(n_episodes))
		file_log.write("n_steps = {}\n".format(n_steps))
		file_log.write("do_reset_env = {}\n".format(do_reset_env))
	
	if ddpg_load:
		agent.load()
		
	global_step_number = 0
	
	for episode in range(n_episodes):
		#print("** rank {}, episode {}".format(rank,episode))
		if do_reset_env:
			#print("RESET !")
			env.reset_env()
		
		state = env.reset()
			
		if (args.gui):
		   env.draw_env_box()
		   
		noise.reset()
		episode_reward = 0
		for step in range(n_steps):
			action = agent.get_action(state)
			action = noise.get_action(action, step)
			new_state, reward, done, info = env.step(action) 
			agent.memory.push(state, action, reward, new_state, done)
			global_step_number += 1

			if len(agent.memory) > ddpg_batch_size:
				agent.update(ddpg_batch_size)

			state = new_state
			episode_reward += reward
			   

	   
		#print('[{}] rank is: {}, episode is: {}, episode reward is: {:.3f}'.format(datetime.now(), rank, episode, episode_reward))
		
		
		global_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.SUM)/MPI.COMM_WORLD.Get_size()
		list_global_rewards.append(global_reward)
		
		if rank == 0:
			print('=> [{}] episode is: {}, eval success rate is: {:.3f}'.format(datetime.now(), episode, list_global_rewards[episode]))
			file_log.write('=> [{}] episode is: {}, eval success rate is: {:.3f}\n'.format(datetime.now(), episode, list_global_rewards[episode])) 
			
			if episode % ddpg_log_interval == 0:
				agent.save()
			  
	if rank == 0:          
	   agent.save()
	   print("end mode train !")
	   print("time elapsed = {}".format(datetime.now()-start))
	   file_log.write("end mode train !\n")
	   file_log.write("time elapsed = {}\n".format(datetime.now()-start))
	   file_log.close()
		
		
	
        
if __name__ == '__main__':
	main()

