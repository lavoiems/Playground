#!/bin/bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
export PYTHONPATH=''
echo Running on $HOSTNAME

python main.py --server http://samuel.zapto.org --port 14445 "$@"
