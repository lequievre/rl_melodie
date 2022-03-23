job_rl01.sh -> Submitted batch job 41372014 


mpirun -n 32 --use-hwthread-cpus --oversubscribe python main.py --max_episode 63
 --max_step 300 --log_interval 10 --save_dir_name './w32/' --load_db_dir_name '/
db1/'

-> change p.DIRECT into env

self.distance_threshold=0.05

end mode train !
time elapsed = 3:19:11.398621

-> test (without gui)
python main.py --mode test --load_db_dir_name '/db1/'
-> test (with gui)
python main.py --mode test --load_db_dir_name '/db1/' --gui True


mean distance error = 0.04852432738989591
sum distance error = 4.852432738989592


python main.py --mode show_database --load_db_dir_name '/db1/'

****** CONFIG DATABASE ***************
nb_x=5, nb_y=20, nb_z=5
d_x=0.20000004768371582, d_y=0.6000000238418579, d_z=0.25
step_x=0.0333333412806193, step_y=0.02857142970675514, step_z=0.041666666666666664
range_x=6, range_y=21, range_z=6
delta_x=5, delta_y=11
**************************************



 
