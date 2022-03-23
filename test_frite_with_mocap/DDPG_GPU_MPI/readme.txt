=> version with 2 points of 'frite' to track

for train :
mpirun -n 4 python main.py --cuda True --max_episode 10


for test :
-> change p.DIRECT line 44 file xarm_reach_env.py
python main.py --mode test

