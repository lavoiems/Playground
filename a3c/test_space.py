import subprocess
import time
for i in range(1, 101, 2):
    lr = i / 1e4
    print(lr)
    subprocess.Popen(['sbatch', '-c', '4', '--mem=4000', '--time=719', '--account=def-bengioy', 'start.sh', '--exp-name', 'a3c_space_sn_actor_depth_3_lr_%s' % lr, '--use-sn-actor', 'True', '--num-processes', '16', '--depth', '3', '--num-episodes', '400', '--lr', str(lr), '--root-path', '/home/lavoiems/scratch/a3c', '--env-name', 'SpaceInvaders-v4'])
    time.sleep(5)
