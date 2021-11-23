import json
import requests
import os
import os.path
import sys

if len(sys.argv) != 2:
	print('Usage: '+sys.argv[0]+' INTXT')
	print('INTXT = tsv file with columns PDB, UniProtID, UniProtName (e.g. 1tqn P08684 CP3A4_HUMAN)')
	exit()
table_file = sys.argv[1]

with open(table_file, 'r') as f:
	table = [line.strip().split('\t') for line in f.readlines()]

table.sort(key=lambda t: (t[2], t[0]))

last_name = None

for pdb, uniid, name, *_ in table:
	if name != last_name:
		print(name, uniid, '', sep='\t')
	print('', '', pdb, sep='\t')
	last_name = name