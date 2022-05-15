#!/bin/bash

set -e

CONFIG="Debug";

for ARG in $@; do
    if [ "$ARG" = "--release" ]; then
        CONFIG="Release";
    fi;
done;

docker run --volume $PWD:/app bitnami/dotnet-sdk:6 dotnet build -c $CONFIG

