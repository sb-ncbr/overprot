#!/bin/bash

DIR=$(dirname $0)
cd $DIR

sudo echo  # check sudo before actually doing something

for ARG in $@; do
    if [ "$ARG" = "--clean" ]; then
        rm -rf venv/;
    fi;
done;

sudo apt install python3-venv
python3 -m venv venv/
. venv/bin/activate
pip install -r requirements.txt

sudo apt install nginx redis-server
