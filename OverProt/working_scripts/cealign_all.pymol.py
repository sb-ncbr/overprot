import json
import os
from os import path
import sys
from pymol import cmd
cmd.set('cif_use_auth', False)

# How to run: pymol -qcr cealign_all.py -- CONSENSUS SAMPLE_FILE IN_DIRECTORY OUT_DIRECTORY

CONSENSUS = 'consensus'
PROTEIN = 'sample'

if len(sys.argv) != 5:
	print('Usage: '+sys.argv[0]+' CONSENSUS SAMPLE_FILE IN_DIRECTORY OUT_DIRECTORY')
	print(len(sys.argv))
	exit()
_, consensus_file, sample_file, in_directory, out_directory = sys.argv

with open(sample_file) as f:
	domains = json.load(f)

cmd.load(consensus_file, CONSENSUS)

if not os.path.exists(out_directory):
	os.makedirs(out_directory)

for pdb, domain, chain, rang in domains:
	print(domain)  # debug
	in_file = path.join(in_directory, domain + '.cif')
	out_file = path.join(out_directory, domain + '.cif')
	cmd.load(in_file, PROTEIN)
	try:
		cmd.cealign(CONSENSUS, PROTEIN)
	except pymol.CmdException:
		print('Warning: "cealign" failed, using "super"')
		cmd.super(PROTEIN, CONSENSUS)
	cmd.save(out_file, PROTEIN)
	cmd.delete(PROTEIN)

cmd.quit()