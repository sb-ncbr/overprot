#!/bin/bash
set -e

echo "This script is just for the case that dotnet (outside podman/docker) cannot be installed properly"

# CONFIG="Debug";

# for ARG in $@; do
#     if [ "$ARG" = "--release" ]; then
#         CONFIG="Release";
#     fi;
# done;

# podman run --volume $PWD:/app bitnami/dotnet-sdk:6 dotnet build -c $CONFIG

