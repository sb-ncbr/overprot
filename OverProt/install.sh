
for ARG in $@; do
    if [ "$ARG" = "--clean" ]; then
        rm -rf venv/;
    fi;
done;

sudo apt-get update -y

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

# Install .NET (for StructureCutter and SecStrAnnotator)
dotnet --info || ./dotnet-install.sh -c 3.1

