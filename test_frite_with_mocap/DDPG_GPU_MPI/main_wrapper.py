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
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--gui', default=False, type=bool) # use gui to see graphic results
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)

args = parser.parse_args()

def main():

	if not os.path.isfile(args.config_file):
			raise RuntimeError("=> Config file JSON to load does not exit : " + args.config_file)
			return
	
	json_decoder = JsonDecoder(args.config_file)

	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'

	env_name = json_decoder.config_data["env"]["name"]
	env_random_seed = json_decoder.config_data["env"]["random_seed"]
	env_time_set_action = json_decoder.config_data["env"]["time_set_action"]
	
	print("** ENV PARAMETERS **")
	print("env_name = {}".format(env_name))
	print("env_random_seed = {}".format(env_random_seed))
	print("env_time_set_action = {}".format(env_time_set_action))
	
	ddpg_cuda = json_decoder.config_data["ddpg"]["cuda"]
	ddpg_max_memory_size = json_decoder.config_data["ddpg"]["max_memory_size"]
	ddpg_batch_size = json_decoder.config_data["ddpg"]["batch_size"]
	ddpg_log_interval = json_decoder.config_data["ddpg"]["log_interval"]
	
	print("** DDPG PARAMETERS **")
	print("ddpg_cuda = {}".format(ddpg_cuda))
	print("ddpg_batch_size = {}".format(ddpg_batch_size))
	print("ddpg_max_memory_size = {}".format(ddpg_max_memory_size))
	print("ddpg_log_interval = {}".format(ddpg_log_interval))

	log_name = json_decoder.config_data["log"]["name"]
	database_name = json_decoder.config_data["database"]["name"]
	
	print("** LOG PARAMETERS **")
	print("log_name = {}".format(log_name))
	
	rank = MPI.COMM_WORLD.Get_rank()

	if not os.path.isfile(json_decoder.config_dir_name  + database_name):
		raise RuntimeError("=> Database file to load does not exit : " + json_decoder.config_dir_name  + database_name)
		return
		    
	if rank == 0:
		file_log = open(json_decoder.config_dir_name + log_name, "w+")
		
	env_pybullet = Environment(json_decoder=json_decoder, gui=args.gui)
	env_pybullet.reset()
	db = Database_Frite(json_decoder=json_decoder)
	env = WrapperPandaFriteEnv(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui))

	env.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	np.random.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	if (ddpg_cuda):
		torch.cuda.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	agent = DDPGagent(ddpg_cuda, env, max_memory_size=ddpg_max_memory_size, directory=json_decoder.config_dir_name)
	noise = OUNoise(env.action_space)
    
	list_global_rewards = []
	
	if args.mode == 'train':
    
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
				file_log.flush()
				
				if episode % ddpg_log_interval == 0:
					agent.save()
				  
		if rank == 0:          
		   agent.save()
		   print("end mode train !")
		   print("time elapsed = {}".format(datetime.now()-start))
		   file_log.write("end mode train !\n")
		   file_log.write("time elapsed = {}\n".format(datetime.now()-start))
		   file_log.close()
		   
	elif args.mode == 'test':
		
		start=datetime.now()
		
		file_log.write("mode test !\n")
		
		print("mode test !")
		
		agent.load()
		n_episodes = json_decoder.config_data["env_test"]["n_episodes"]
		n_steps = json_decoder.config_data["env_test"]["n_steps"]
		do_reset_env = json_decoder.config_data["env"]["do_reset_env"]
		wait_time_sleep_after_draw_env_box = json_decoder.config_data["env_test"]["wait_time_sleep_after_draw_env_box"]
		wait_time_sleep_end_episode = json_decoder.config_data["env_test"]["wait_time_sleep_end_episode"]
		do_episode_hit_return = json_decoder.config_data["env_test"]["do_episode_hit_return"]
		
		file_log.write("** ENV MODE TEST **\n")
		file_log.write("config_file = {}\n".format(args.config_file))
		file_log.write("n_episodes = {}\n".format(n_episodes))
		file_log.write("n_steps = {}\n".format(n_steps))
		file_log.write("do_reset_env = {}\n".format(do_reset_env))
		file_log.write("wait_time_sleep_after_draw_env_box = {}\n".format(wait_time_sleep_after_draw_env_box))
		file_log.write("wait_time_sleep_end_episode = {}\n".format(wait_time_sleep_end_episode))
		file_log.write("do_episode_hit_return = {}\n".format(do_episode_hit_return))
		
		print("** ENV MODE TEST **")
		print("n_episodes = {}".format(n_episodes))
		print("n_steps = {}".format(n_steps))
		print("do_reset_env = {}".format(do_reset_env))
		print("wait_time_sleep_after_draw_env_box = {}".format(wait_time_sleep_after_draw_env_box))
		print("wait_time_sleep_end_episode = {}".format(wait_time_sleep_end_episode))
		print("do_episode_hit_return = {}".format(do_episode_hit_return))
				
		nb_dones = 0
		sum_distance_error = 0
		for episode in range(n_episodes):
			print("Episode : {}".format(episode))
			file_log.write("Episode : {}\n".format(episode))
		   
			if do_reset_env:
				print("RESET !")
				file_log.write("RESET !\n")
				env.reset_env()
			
			state = env.reset()
				
			if (args.gui):
			   env.draw_env_box()
			   time.sleep(wait_time_sleep_after_draw_env_box)
			   
			current_distance_error = 0
			
			for step in range(n_steps):
				action = agent.get_action(state)
				
				print("action={}".format(action))
				file_log.write("action = {}\n".format(action))
			   
				new_state, reward, done, info = env.step(action)
				current_distance_error = info['max_distance_error']
				if (args.gui):
					env.draw_id_to_follow()
				
				print("step={}, distance_error={}\n".format(step,info['max_distance_error']))
				file_log.write("step={}, distance_error={}\n".format(step,info['max_distance_error']))
				
				#print("step={}, action={}, reward={}, done={}, info={}".format(step,action,reward, done, info))
				state = new_state
			   
				if done:
				   print("done with step={}  !".format(step))
				   file_log.write("done with step={}  !\n".format(step))
				   
				   nb_dones+=1
				   break
				   
			if (args.gui):
				if do_episode_hit_return:
					input("hit return to continue !")
				else:
					time.sleep(wait_time_sleep_end_episode)
			
		   
			sum_distance_error += current_distance_error
		print("time_set_action = {}".format(env_time_set_action))
		print("nb dones = {}".format(nb_dones))
		print("mean distance error = {}".format(sum_distance_error/n_episodes))
		print("sum distance error = {}".format(sum_distance_error))
		print("time elapsed = {}".format(datetime.now()-start))
		
		file_log.write("time_set_action = {}\n".format(env_time_set_action))
		file_log.write("nb dones = {}\n".format(nb_dones))
		file_log.write("mean distance error = {}\n".format(sum_distance_error/n_episodes))
		file_log.write("sum distance error = {}\n".format(sum_distance_error))
		file_log.write("time elapsed = {}\n".format(datetime.now()-start))
		
		file_log.close()
		
		input("hit return !")
		
		      
if __name__ == '__main__':
	main()
