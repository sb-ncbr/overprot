#!/bin/bash

SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

export OVERPROT_PYTHON='/home/adam/Workspace/Python/OverProt/overprot/OverProt/venv/bin/python'
export OVERPROT_PY='/home/adam/Workspace/Python/OverProt/overprot/OverProt/overprot.py'
export VAR_DIR='/server_data/overprot_data'

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
