
import sys
import json
import requests
import re
from os import path
from collections import OrderedDict
from pymol import cmd

def is_valid_pdbid(string):
	return len(string) == 4 and string.isalnum() and string[0].isdigit()

def is_valid_selection(selection):
	try:
		# debug_log('is_valid_selection(' + selection + ') ...')
		cmd.count_atoms(selection)
		# debug_log('is_valid_selection(' + selection + ') Yes')
		return True
	except:
		# debug_log('is_valid_selection(' + selection + ') No')
		return False

def cath(domain_name):
	pdb = domain_name[0:4]
	if not is_valid_pdbid(pdb):
		print(pdb + ' is not valid PDB ID')
		return False
	url = 'http://www.ebi.ac.uk/pdbe/api/mappings/cath/' + pdb
	pdbe_response = json.loads(requests.get(url).text)
	families = pdbe_response[pdb]['CATH']
	for fam, fam_info in families.items():
		mappings = [ mapp for mapp in fam_info['mappings'] if mapp['domain'] == domain_name ]
		if len(mappings) > 0:
			ranges = sorted( (mapp['struct_asym_id'], mapp['start']['residue_number'], mapp['end']['residue_number']) for mapp in mappings )
			chain = ranges[0][0]
			ranges_text = ','.join( str(start)+':'+str(end) for ch, start, end in ranges )
			selection = pdb + ' and chain ' + chain + ' and resi ' + '+'.join( str(start)+'-'+str(end) for ch, start, end in ranges )
			complement = '(bychain ' + domain_name + ') and not ' + domain_name
			ext_complement = 'byres (' + complement + ') extend 1'
			print('CATH ' + fam + '   ' + domain_name + '   ' + pdb + ',' + chain + ',' + ranges_text)

			if not is_valid_selection(pdb):
				cmd.set('cif_use_auth', 0)
				cmd.fetch(pdb, async=0)
			cmd.hide('everything', pdb)
			cmd.select(domain_name, selection)
			cmd.show('ribbon', ext_complement)
			cmd.set('ribbon_width', 1)
			cmd.show('cartoon', domain_name)
			cmd.spectrum(selection=domain_name + ' and symbol C')
			cmd.zoom(domain_name)
			cmd.deselect()
			return True
	print(domain_name + ' not found')
	return False


# The script for PyMOL ########################################################

# test()
cmd.extend('cath', cath)
print('Command "cath" has been added. Example usage: cath 1tqnA00.')


