#!/bin/bash

DIR=$(dirname $0)
DC="$DIR/docker-compose.yaml"

echo $DC
echo
echo '>>> DOWN <<<'
docker-compose -f $DC down
echo
echo '>>> BUILD <<<'
docker build . -t overprot-server:0.9 -t registry.gitlab.com/midlik/overprot
echo
echo '>>> UP <<<'
docker-compose -f $DC up -d
sleep 4
echo
echo '>>> LOG <<<'
docker logs overprot_overprot_server_1
