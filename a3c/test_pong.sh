#!/usr/bin/env bash

for d in 0 1 2 3 4 5; do
  sbatch -c 4 --mem=1000 start.sh --exp-name a3c_pong_sn_actor_depth_${d}_lr_0.001 --use-sn-actor True --num-processes 16 --depth ${d} --num-episodes 400 --lr 0.001 --root-path /data/milatmp1/lavoiems/a3c
  sleep 1
done

for d in 0 1 2 3 4 5; do
  sbatch -c 4 --mem=1000 start.sh --exp-name a3c_pong_sn_critic_depth_${d}_lr_0.001 --use-sn-critic True --num-processes 16 --depth ${d} --num-episodes 400 --lr 0.0001 --root-path /data/milatmp1/lavoiems/a3c
  sleep 1
done

for d in 0 1 2 3 4 5; do
  sbatch -c 4 --mem=1000 start.sh --exp-name a3c_pong_no_sn_depth_${d}_lr_0.001 --num-processes 16 --depth ${d} --num-episodes 400 --lr 0.0001 --root-path /data/milatmp1/lavoiems/a3c
  sleep 1
done
