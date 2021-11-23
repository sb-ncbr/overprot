#!/bin/bash

SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

. ./config.sh

sudo nginx -s quit 2> /dev/null

# Kill Gunicorn and RedisQueue workers

PROC_DIR="$VAR_DIR/running_processes"

for PID in $(ls $PROC_DIR) 
do 
    sudo kill -9 $PID
done

rm -rf $PROC_DIR

# Print started processes
echo 'Please wait 5 seconds, shutting down processes'
sleep 5
echo 'STILL RUNNING PROCESSES:'
ps -A | grep nginx
ps -A | grep gunicorn:
ps -A | grep rq:

