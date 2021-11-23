#!/bin/bash

DIR=$(dirname $0)
cd $DIR

sudo echo  # check sudo before actually doing something

sudo apt-get update -y
sudo apt-get install -y python3-venv

sudo apt-get install -y nginx
sudo apt-get install -y redis-server
sudo apt-get install -y gettext-base   # for envsubst
