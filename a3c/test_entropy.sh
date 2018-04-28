#!/usr/bin/env bash

for d in 0.001 0.01 0.1 1 10; do
    sbatch -c 4 --mem=8000 start.sh --exp-name a3c_3layers_sn_actor_space_steps_200_entropy_${d} --use-sn-actor True --max-grad-norm 0 --num-processes 16 --env-name SpaceInvaders-v4 --depth 3 --entropy-coef ${d} --num-steps 200
    sleep 1
done

for d in 0.001 0.01 0.1 1 10; do
    sbatch -c 4 --mem=8000 start.sh --exp-name a3c_3layers_sn_both_space_steps_200_entropy_${d} --use-sn-critic True --use-sn-actor True --max-grad-norm 0 --num-processes 16 --env-name SpaceInvaders-v4 --depth 3 --entropy-coef ${d} --num-steps 200
    sleep 1
done
