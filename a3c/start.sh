#!/bin/bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
echo Running script $1

python main.py --server http://samuel.zapto.org --port 14445 --root-path /data/milatmp1/lavoiems/a3c "$@"
