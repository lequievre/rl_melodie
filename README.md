Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Mélodie HANI DANIEL ZAKARIA
PhD Student
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.hani_daniel_zakaria@uca.fr



# How to install virtualenv on ubuntu 20.04


virtualenv is a tool to create lightweight “virtual environments” with their own site directories isolated from system site directories.
Each "virtual environment" has its own Python binary (which matches the version of the binary that was used to create this environment) 
and can have its own independent set of installed Python packages in its site directories.


sudo apt install python3-pip
pip3 install virtualenv

How to create a virtualenv named 've_rl' and activate it :
--------------------------------------------------------

virtualenv ve_rl --python=python3
source ve_rl/bin/activate

Then necessary python3 packages can be installed into that virtualenv :
---------------------------------------------------------------------

pip install --upgrade pip

pip install gym
pip install torch
pip install matplotlib
pip install mpi4py
pip install pybullet


Details of 'main.py' parameters :
-------------------------------
n : number of mpi processors requested.
gui : boolean to show or not the gui.
max_episode : number of episodes.
max_step : number of episode steps.
log_interval : frequency of saving neural weights, every 'log_interval' episodes.
max_memory_size : size of DDPG Agent memory.
batch_size : size of the trained tuples.
distance_threshold : distance necessary to success the goal.
save_dir_name : name of neural weights directory (create it when it doesn't exist).
generate_database_name : name of generated 'goal' database (by default 'database_id_frite.txt').
load_database_name : name of loaded 'goal' database (by default 'database_id_frite.txt').
load_db_dir_name : directory name of 'goal' database loaded.
db_nb_x : how to divide the 'goal space' on x to generate a 'goal' database.
db_nb_y : how to divide the 'goal space' on y to generate a 'goal' database.
db_nb_z : how to divide the 'goal space' on z to generate a 'goal' database.
random_seed : value to initialize the random number generator
generate_db_dir_name : directory name of generated database 

How to train :
------------
cd DDPG_GPU_MPI
mpirun -n 32 python main.py --max_episode 63 --max_step 300 --log_interval 10 --save_dir_name './w32/' --load_db_dir_name '/extra_small/'

The database used is a file named 'database_id_frite.txt' by default in the directory 'DDPG_GPU_MPI/databases/extra_small'
The neural network weights will be saved in the directory 'DDPG_GPU_MPI/w32'

How to test :
-----------
cd DDPG_GPU_MPI
python main.py --mode test --load_db_dir_name '/extra_small/' --gui True --save_dir_name './w32/'

How to generate a database :
--------------------------
cd DDPG_GPU_MPI
python main.py --mode generate_database --generate_db_dir_name '/extra_small/'  --db_nb_x 5 --db_nb_y 20 --db_nb_z 5

Create a file named 'database_id_frite.txt' in the directory 'DDPG_GPU_MPI/databases/extra_small'.
