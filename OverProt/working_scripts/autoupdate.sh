PYTHON="/home/adam/Workspace/Python/OverProt/overprot/OverProt/venv/bin/python"
OVERPROT_MULTI="/home/adam/Workspace/Python/OverProt/overprot/OverProt/overprot_multifamily.py"
DATA_DIR="/home/adam/Workspace/Python/OverProt/data/multifamily"

if [ $# -lt 1 ] ; then 
	echo '1 arguments required:  UPDATE_ID  (e.g. all_unique_pdb-20210217-try1)'  1>&2;
	return 1;
	fi
UPDATE=$1

UPDATE_DIR=$DATA_DIR/$UPDATE

CONFIG=$(dirname $0)/overprot-config-overprotserverdb.ini

mkdir -p $UPDATE_DIR
echo "Data will be in $UPDATE_DIR"
$PYTHON $OVERPROT_MULTI --download_family_list_by_size  --config $CONFIG --collect  xxx all $UPDATE_DIR/  > $UPDATE_DIR/stdout.txt  2> $UPDATE_DIR/stderr.txt

