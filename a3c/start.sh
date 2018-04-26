#!/bin/bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

module load miniconda3
source activate py36

python main.py --server http://samuel.zapto.org --port 14445 "$@"
