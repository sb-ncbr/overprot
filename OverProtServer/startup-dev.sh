#!/bin/bash

SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

. ./config-dev.sh

. venv/bin/activate

# CHECK
python -c 'import overprot_server'

# REDIS-QUEUE
rq worker $RQ_QUEUE  & # run repeatedly to start more workers

# FLASK
export FLASK_APP='overprot_server'
export FLASK_ENV='development'
flask run -p 8080
