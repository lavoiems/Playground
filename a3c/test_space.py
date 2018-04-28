import subprocess
import time
for i in range(1, 101, 5):
    lr = i / 1e4
    print(lr)
    subprocess.Popen(['sbatch', '-c', '4', '--mem=4000', 'start.sh', '--exp-name', 'a3c_space_sn_actor_depth_3_lr_%s' % lr, '--use-sn-actor', 'True', '--num-processes', '8', '--depth', '3', '--num-episodes', '400', '--lr', str(lr), '--root-path', '/data/milatmp1/lavoiems/a3c', '--env-name', 'SpaceInvaders-v4'])
    time.sleep(1)
