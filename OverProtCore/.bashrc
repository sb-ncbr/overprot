. ~/.bashrc
. venv/bin/activate  \
&&  { echo "Activated virtual environment: $VIRTUAL_ENV"; echo; } \
|| { echo "Failed to activate virtual environment!"; echo "Please create it by running 'sh install.sh' and restart this shell."; echo; }
