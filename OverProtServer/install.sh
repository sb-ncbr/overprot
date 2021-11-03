#!/bin/bash

DIR=$(dirname $0)
cd $DIR

sudo echo  # check sudo before actually doing something

for ARG in $@; do
    if [ "$ARG" = "--clean" ]; then
        rm -rf venv/;
    fi;
done;

sudo apt-get update -y
sudo apt-get install -y python3-venv
python3 -m venv venv/
. venv/bin/activate
python3 -m pip install -r requirements.txt

sudo apt-get install -y nginx
sudo apt-get install -y redis-server
sudo apt-get install -y gettext-base   # for envsubst
