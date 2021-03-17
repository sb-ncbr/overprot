import json
import requests
import os
import os.path
import sys

if len(sys.argv) < 3:
	print('Usage: '+sys.argv[0]+' INJSON OUTJSON')
	exit()
pdb_list_file = sys.argv[1]
out_file = sys.argv[2]

def enumerate(name, elements):
	print(name + ' [' + str(len(elements)) + ']: '  + ', '.join(elements) + '\n')

API_URL='http://www.ebi.ac.uk/pdbe/api/mappings/uniprot/'
UNIPROT='UniProt'
MAPPINGS='mappings'
CHAIN='chain_id'

with open(pdb_list_file, 'r') as f:
	pdb_list=json.load(f)

result={}

for pdb in pdb_list:
	#print(pdb)
	r = json.loads(requests.get(API_URL + pdb).text)
	unip = r[pdb][UNIPROT]
	domains = pdb_list[pdb]
	chains = [chain for dom_name, chain, rang in domains]
	codes = set( code for code in unip for mapp in unip[code][MAPPINGS] if mapp[CHAIN] in chains )
	if len(codes) > 1:
		print('WARNING: PDB ID '+pdb+' maps to multiple UniProt IDs: '+', '.join(sorted(codes)))
	elif len(unip) == 0:
		print('WARNING: PDB ID '+pdb+' maps to no UniProt ID')
		continue
	else:
		uniid = sorted(codes)[0]
		if uniid not in result:
			result[uniid] = {}
		result[uniid][pdb] = pdb_list[pdb]

with open(out_file, 'w') as f:
	json.dump(result, f, indent=4, sort_keys=True)
