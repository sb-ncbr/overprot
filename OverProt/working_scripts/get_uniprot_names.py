import json
import requests
import os
import os.path
import sys

if len(sys.argv) != 2:
	print('Usage: '+sys.argv[0]+' INTXT')
	exit()
pdb_list_file = sys.argv[1]

API_URL='http://www.ebi.ac.uk/pdbe/api/mappings/uniprot/'

with open(pdb_list_file, 'r') as f:
	pdb_list = [line.strip() for line in f.readlines()]

for pdb in pdb_list:
	r = json.loads(requests.get(API_URL + pdb).text)
	uniprot = r[pdb]['UniProt']
	if len(uniprot) != 1:
		sys.stderr.write(f'WARNING: {pdb} maps to multiple UniProtIDs (' + ', '.join(uniprot.keys()) + ')\n')
	for uni, annot in uniprot.items():
		print(pdb, uni, annot['identifier'], annot['name'], sep='\t')