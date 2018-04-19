#!/bin/bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
echo Running script $1
export OMP_NUM_THREADS=8

python main.py --server http://samuel.zapto.org --port 14445

