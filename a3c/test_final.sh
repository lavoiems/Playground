#!/usr/bin/env bash

#for seed in 0 100 1000 10000 100000; do
#  sbatch -c 4 --mem=1000 start.sh --exp-name final_a3c_space_sn_actor_seed_${seed} --use-sn-actor True --num-processes 16 --depth-actor 3 --num-episodes 200 --lr 0.0001 --root-path /data/milatmp1/lavoiems/a3c --env-name SpaceInvaders-v4 --seed ${seed}
#  sleep 1
#done

for seed in 0 100 1000 10000; do
  sbatch -c 4 --mem=1000 start.sh --exp-name final_a3c_space_vanilla_seed_${seed} --num-processes 16 --num-episodes 200 --lr 0.0001 --root-path /data/milatmp1/lavoiems/a3c --env-name SpaceInvaders-v4 --seed ${seed}
  sleep 1
done

