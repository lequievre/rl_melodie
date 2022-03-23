job_rl01.sh -> Submitted batch job 41372014 

-> same as small db, into 'db1' directory.
mpirun -n 32 --use-hwthread-cpus --oversubscribe python main.py --max_episode 63
 --max_step 300 --log_interval 10 --save_dir_name './w32/' --load_db_dir_name '/
db1/'

-> change p.DIRECT into env

self.distance_threshold=0.05

end mode train !
time elapsed = 3:19:11.398621

-> test (without gui)
python main.py --mode test
-> test (with gui)
python main.py --mode test --gui True


mean distance error = 0.04908861195668578
sum distance error = 4.908861195668578

-> use 'medium' database into 'default_load' directory
python main.py --mode show_database

****** CONFIG DATABASE ***************
nb_x=8, nb_y=22, nb_z=10
d_x=0.20000004768371582, d_y=0.7999999523162842, d_z=0.2999999523162842
step_x=0.02222222752041287, step_y=0.034782606622447136, step_z=0.027272722937844017
range_x=9, range_y=23, range_z=11
delta_x=7, delta_y=12
**************************************




 
