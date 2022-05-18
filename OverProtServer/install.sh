#!/bin/bash
set -e

DIR=$(dirname $0)
cd $DIR

CLEAN='false'
SUDO='sudo'

for ARG in $@; do
    if [ "$ARG" = "--clean" ]; then CLEAN='true'; fi;
    if [ "$ARG" = "--no-sudo" ]; then SUDO=''; fi;
done;

$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends python3-venv
$SUDO apt-get install -y --no-install-recommends nginx
$SUDO apt-get install -y --no-install-recommends redis-server
$SUDO apt-get install -y --no-install-recommends gettext-base   # for envsubst

if [ "$CLEAN" = "true" ]; then rm -rf venv/; fi;
python3 -m venv venv/
. venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip freeze > requirements-actual.txt
