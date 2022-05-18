#!/bin/bash

# This installation script is designed to work on Ubuntu (version 20.04).
# If it is not compatible with your operating system, 
# try using OverProt via Docker instead (see README.md for details).

set -e

SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

CLEAN='false'
SUDO='sudo'

for ARG in $@; do
    if [ "$ARG" = "--clean" ]; then CLEAN='true'; fi;
    if [ "$ARG" = "--no-sudo" ]; then SUDO=''; fi;
done;


# Install essential basic packages
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends  curl  ca-certificates  libicu-dev

# Install .NET (for StructureCutter and SecStrAnnotator)
# TODO check runtime version (--list-runtimes)
# TODO install here with --install-dir ./dependencies/dotnet
if dotnet --info; then
    echo "OverProtCore/install.sh: Dotnet already installed"
else
    echo "OverProtCore/install.sh: Trying to install Dotnet..."
    ./dotnet-install.sh -c 6.0
    PATH="$PATH:$HOME/.dotnet"
    echo -e '\n# Add Dotnet to PATH:\nPATH="$PATH:$HOME/.dotnet"' >> $HOME/.bashrc
fi

if dotnet --info; then
    echo "OverProtCore/install.sh: Dotnet successfully installed"
else
    echo "OverProtCore/install.sh: Dotnet installation failed"
    exit 1
fi

# Create Python virtual environment and install Python packages in it
$SUDO apt-get install -y --no-install-recommends python3-venv
if [ "$CLEAN" = "true" ]; then rm -rf venv/; fi;
python3 -m venv venv/
. venv/bin/activate
pip3 install -r requirements.txt

# Install PyMOL and make it available in virtual environment - hack (PyMOL cannot be installed through pip)
$SUDO apt-get install -y --no-install-recommends pymol
DIR=$(echo $VIRTUAL_ENV/lib/python3.*/site-packages)
ln -s "/usr/lib/python3/dist-packages/pymol" "$DIR/pymol"
ln -s "/usr/lib/python3/dist-packages/chempy" "$DIR/chempy"
ln -s "/usr/lib/python3/dist-packages/pymol2" "$DIR/pymol2"

