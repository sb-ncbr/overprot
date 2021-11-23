#!/bin/bash
set -e

echo 'startup-docker.sh'

export OVERPROT_PYTHON='/srv/bin/overprot/OverProtCore/venv/bin/python'
export OVERPROT_PY='/srv/bin/overprot/OverProtCore/overprot.py'
export VAR_DIR='/srv/var'
INIT_VAR_DIR='/srv/init_var'
export DATA_DIR='/srv/data'
export SSL_CERT='/srv/ssl/certificate.pem'
export SSL_KEY='/srv/ssl/key.pem'
export RQ_QUEUE='overprot_jobs'
export GUNICORN_PORT='4000'

# N_RQ_WORKERS=8  # to be set in docker ENV
test -z "$N_RQ_WORKERS" && echo "Error: environment variable N_RQ_WORKERS is not set" && exit 1

# N_GUNICORN_WORKERS=4  # to be set in docker ENV
test -z "$N_GUNICORN_WORKERS" && echo "Error: environment variable N_GUNICORN_WORKERS is not set" && exit 1

test -z "$HTTP_PORT" && test -z "$HTTPS_PORT" && echo "Error: at least one of HTTP_PORT, HTTPS_PORT must be set" && exit 1
test -n "$HTTPS_PORT" -a ! -f "$SSL_CERT" && echo "Error: when HTTPS_PORT is set, SSL certificate must be mounted to $SSL_CERT" && exit 1
test -n "$HTTPS_PORT" -a ! -f "$SSL_KEY" && echo "Error: when HTTPS_PORT is set, SSL private key must be mounted to $SSL_KEY" && exit 1

SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

# Preparation
source venv/bin/activate
RQ=$(which rq)  ||  { echo "rq is not correctly installed" && exit 1; }
GUNICORN=$(which gunicorn)  ||  { echo "gunicorn is not correctly installed" && exit 1; }
NGINX=$(which nginx)  ||  { echo "nginx is not correctly installed" && exit 1; }

echo "Starting OverProt Server"
echo "VAR_DIR=$VAR_DIR"

cp -r $INIT_VAR_DIR/* $VAR_DIR

START_TIME=$(date -u +%Y%m%d_%H%M%S)
LOG_DIR="$VAR_DIR/logs/run_$START_TIME"
echo "Logs: $LOG_DIR"
PROC_DIR="$VAR_DIR/running_processes"
rm -rf $PROC_DIR
mkdir -p $PROC_DIR

# Start Gunicorn
echo "Starting Gunicorn"
mkdir -p "$LOG_DIR/gunicorn"
gunicorn -w $N_GUNICORN_WORKERS -b 127.0.0.1:$GUNICORN_PORT overprot_server:app > "$LOG_DIR/gunicorn/out.txt" 2> "$LOG_DIR/gunicorn/err.txt" & 
touch $PROC_DIR/$!

# Start RedisQueue workers
echo "Starting RedisQueue"
mkdir -p "$LOG_DIR/rq"
service redis-server start
for RQ_WORKER in $(seq -w 1 $N_RQ_WORKERS)
do
    $RQ worker $RQ_QUEUE > "$LOG_DIR/rq/worker_$RQ_WORKER.out.txt" 2> "$LOG_DIR/rq/worker_$RQ_WORKER.err.txt" & 
    touch $PROC_DIR/$!
done

# Start Nginx
echo "Starting Nginx"
export NGINX_LOG_DIR="$LOG_DIR/nginx"
mkdir -p $NGINX_LOG_DIR
NGINX_CONF_DIR="$VAR_DIR/nginx_conf"
mkdir -p $NGINX_CONF_DIR
cp -r $SW_DIR/nginx/* $NGINX_CONF_DIR/
test -n "$HTTP_PORT" && export MAYBE_INCLUDE_HTTP="include $NGINX_CONF_DIR/nginx-http.conf;"
test -n "$HTTPS_PORT" && export MAYBE_INCLUDE_HTTPS="include $NGINX_CONF_DIR/nginx-https.conf;"
ENVS='$NGINX_LOG_DIR $SSL_CERT $SSL_KEY $HTTP_PORT $HTTPS_PORT $GUNICORN_PORT $MAYBE_INCLUDE_HTTP $MAYBE_INCLUDE_HTTPS'
for F in $NGINX_CONF_DIR/*.template.conf; do envsubst "$ENVS" < $F > ${F%.template.conf}.conf; done
NGINX_CONF=$NGINX_CONF_DIR/nginx.conf
sudo nginx -c $NGINX_CONF

# Print some logs:
echo "N_GUNICORN_WORKERS=$N_GUNICORN_WORKERS"
echo "N_RQ_WORKERS=$N_RQ_WORKERS"
echo "HTTP_PORT=$HTTP_PORT"
echo "HTTPS_PORT=$HTTPS_PORT"
echo "OVERPROT_STRUCTURE_SOURCE=$OVERPROT_STRUCTURE_SOURCE"

sleep 2
echo 'Started processes:'
ps -A | grep 'gunicorn' || echo 'Warning: no "gunicorn" running'
ps -A | grep 'redis' || echo 'Warning: no "redis" running'
ps -A | grep 'rq' || echo 'Warning: no "rq" running'
ps -A | grep 'nginx' || echo 'Warning: no "nginx" running'

echo 'Starting OverProt Server complete'

sleep infinity
