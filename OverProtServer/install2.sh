#!/bin/bash

DIR=$(dirname $0)
cd $DIR

sudo echo  # check sudo before actually doing something

for ARG in $@; do
    if [ "$ARG" = "--clean" ]; then
        rm -rf venv/;
    fi;
done;

python3 -m venv venv/
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip freeze > requirements-actual.txt
