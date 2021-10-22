#!/bin/bash

export OVERPROT_PYTHON='/server/overprot/software/overprot/OverProt/venv/bin/python'
export OVERPROT_PY='/server/overprot/software/overprot/OverProt/overprot.py'
export ROOT_DIR='/server/overprot/data'
NGINX_DIR='/server/overprot/data/nginx'
export RQ_QUEUE='overprot_jobs'

# N_RQ_WORKERS=8  # to be set in docker ENV
test -z "$N_RQ_WORKERS" && echo "Error: environment variable N_RQ_WORKERS is not set" && exit 1

# N_GUNICORN_WORKERS=4  # to be set in docker ENV
test -z "$N_GUNICORN_WORKERS" && echo "Error: environment variable N_GUNICORN_WORKERS is not set" && exit 1
GUNICORN_PORT='4000'


SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

# Preparation
. venv/bin/activate
RQ=$(which rq)
GUNICORN=$(which gunicorn)
NGINX=$(which nginx)

echo "STARTING OVERPROT SERVER"
echo "ROOT_DIR: $ROOT_DIR"

START_TIME=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT_DIR/logs/run_$START_TIME"
echo "LOGS: $LOG_DIR"
PROC_DIR="$ROOT_DIR/running_processes"
rm -rf $PROC_DIR
mkdir -p $PROC_DIR

# Start Gunicorn
mkdir -p "$LOG_DIR/gunicorn"
gunicorn -w $N_GUNICORN_WORKERS -b 127.0.0.1:$GUNICORN_PORT overprot_server:app > "$LOG_DIR/gunicorn/out.txt" 2> "$LOG_DIR/gunicorn/err.txt" & 
touch $PROC_DIR/$!

# Start RedisQueue workers
mkdir -p "$LOG_DIR/rq"
for RQ_WORKER in $(seq -w 1 $N_RQ_WORKERS)
do
    $RQ worker $RQ_QUEUE > "$LOG_DIR/rq/worker_$RQ_WORKER.out.txt" 2> "$LOG_DIR/rq/worker_$RQ_WORKER.err.txt" & 
    touch $PROC_DIR/$!
done

# Start Nginx
NGINX_CONF_TEMPLATE="$SW_DIR/nginx/nginx-docker.template.conf"
NGINX_CONF="$SW_DIR/nginx/nginx-docker.conf"
NGINX_LOG_DIR="$NGINX_DIR/logs/run_$START_TIME"
mkdir -p $NGINX_LOG_DIR
sed "s:{{NGINX_LOG_DIR}}:$NGINX_LOG_DIR:g" $NGINX_CONF_TEMPLATE > $NGINX_CONF
sudo nginx -s stop 2> /dev/null
sudo nginx -c $NGINX_CONF

# Print started processes
sleep 2
echo 'STARTED PROCESSES:'
ps -A | grep nginx
ps -A | grep gunicorn
ps -A | grep ' rq'

sleep infinity
