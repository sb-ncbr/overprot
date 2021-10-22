#!/bin/bash

SW_DIR=$(realpath $(dirname $0))
cd $SW_DIR

for ARG in $@; do
    if [ "$ARG" = "--clean" ]; then
        rm -rf venv/;
    fi;
done;

sudo apt-get update -y

# Install .NET (for StructureCutter and SecStrAnnotator)
if dotnet --info; then
    echo "OverProt/install.sh: Dotnet already installed"
else
    echo "OverProt/install.sh: Trying to install Dotnet..."
    sudo apt-get install -y curl
    sudo apt-get install -y libicu-dev
    ./dotnet-install.sh -c 3.1
    PATH="$PATH:$HOME/.dotnet"
    echo -e '\n# Add Dotnet to PATH:\nPATH="$PATH:$HOME/.dotnet"' >> $HOME/.bashrc
fi

if dotnet --info; then
    echo "OverProt/install.sh: Dotnet successfully installed"
else
    echo "OverProt/install.sh: Dotnet installation failed"
    exit 1
fi

# Python virtual environment
sudo apt-get install -y python3-venv
python3 -m venv venv/
. venv/bin/activate
pip install -r requirements.txt

# Hack (PyMOL cannot be installed through pip)
sudo apt-get install -y pymol
DIR=$(echo $VIRTUAL_ENV/lib/python3.*/site-packages)
ln -s "/usr/lib/python3/dist-packages/pymol" "$DIR/pymol"
ln -s "/usr/lib/python3/dist-packages/chempy" "$DIR/chempy"
ln -s "/usr/lib/python3/dist-packages/pymol2" "$DIR/pymol2"

