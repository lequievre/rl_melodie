import time
import os
import gym
import numpy as np
import gym_xarm
"""
from gym.envs.registration import register

env_dict = gym.envs.registration.registry.env_specs.copy()

for env in env_dict:
    print(env)
    if 'Simple_XarmReach-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]


register(
    id='Simple_XarmReach-v0',
    entry_point='gym_xarm.envs:XarmReachEnv',
)

"""
#env = gym.make('XarmPDPickAndPlace-v0')
env = gym.make('Simple_XarmReach-v0')

env.reset()
env.render()
#env.printAllInfo()
input("taper return !")
for i in range(5):
   action = env.action_space.sample()
   action[0] = 0.1
   action[1] = 0.0
   action[2] = 0.0
   action[3] = 0.0
   env.step(action)
   env.render()
   input("taper return !")
   
env.close()


"""
# FetchPickAndPlace-v0 XarmPDStackTower-v0 XarmPDPushWithDoor-v0 XarmPDOpenBoxAndPlace-v0 XarmPDHandover-v0
env = gym.make('XarmReach-v0')
agent = lambda ob: env.action_space.sample()
ob = env.reset()
for _ in range(env._max_episode_steps*100):
    env.render()
    assert env.observation_space.contains(ob)
    a = agent(ob)
    assert env.action_space.contains(a)
    (ob, _reward, done, _info) = env.step(a)
    time.sleep(0.02)
env.close()
"""
