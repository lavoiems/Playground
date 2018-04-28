#!/usr/bin/env bash

for seed in 0 100 1000; do
  for d in 0 3 5; do
    sbatch -c 4 --mem=1000 start.sh --exp-name final_a3c_pong_sn_actor_depth_${d}_seed_${seed} --use-sn-actor True --num-processes 16 --depth-actor ${d} --depth-critic ${d} --num-episodes 400 --lr 0.0001 --root-path /data/milatmp1/lavoiems/a3c --seed ${seed}
    sleep 1
  done
done

for seed in 0 100 1000; do
  for d in 0 3 5; do
    sbatch -c 4 --mem=1000 start.sh --exp-name final_a3c_pong_sn_critic_depth_${d}_seed_${seed} --use-sn-critic True --num-processes 16 --depth-actor ${d} --depth-critic ${d} --num-episodes 400 --lr 0.0001 --root-path /data/milatmp1/lavoiems/a3c --seed ${seed}
    sleep 1
  done
done


for seed in 0 100 1000; do
  for d in 0 3 5; do
    sbatch -c 4 --mem=1000 start.sh --exp-name final_a3c_pong_no_sn_depth_${d}_seed_${seed} --num-processes 16 --depth-actor ${d} --depth-critic ${d} --num-episodes 400 --lr 0.0001 --root-path /data/milatmp1/lavoiems/a3c --seed ${seed}
    sleep 1
  done
done

