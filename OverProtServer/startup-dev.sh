#!/bin/bash

set -e

SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

# Set the following paths according to your system:
export OVERPROT_PYTHON='/home/adam/Workspace/Python/OverProt/overprot/OverProtCore/venv/bin/python'
export OVERPROT_PY='/home/adam/Workspace/Python/OverProt/overprot/OverProtCore/overprot.py'
export VAR_DIR='/home/adam/Workspace/Python/OverProt/docker_mount/var'
export DATA_DIR='/home/adam/Workspace/Python/OverProt/docker_mount/data'

export RQ_QUEUE='overprot_jobs'
export N_RQ_WORKERS=8

source venv/bin/activate

# CHECK
python -c 'import overprot_server'

# REDIS-QUEUE
rq worker $RQ_QUEUE  & # run repeatedly to start more workers

# FLASK
export FLASK_APP='overprot_server'
export FLASK_ENV='development'
flask run -p 8081
