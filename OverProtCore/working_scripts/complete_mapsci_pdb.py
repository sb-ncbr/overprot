# Performs clustering of SSEs from a set of domains, preserving SSE order and type.
# Requires these files to be in the working directory:
#     cytos.exe
#     script_align.py

from pathlib import Path
import sys
import numpy as np
import argparse
import itertools
import heapq
import re

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('input_pdb', help='Consensus PDB from MAPSCI', type=Path)
args = parser.parse_args()

with open(args.input_pdb) as f:
	for line in iter(f.readline, ''):
		if len(line) >= 6 and (line[0:4] == 'ATOM' or line[0:6] == 'HETATM'):
			changed = line[0:21] + 'A' + line[22:54] + '  1.00  0.00           C  \n'  # add chain ID and stuff like occupancy etc.
			sys.stdout.write(changed)
		else:
			sys.stdout.write(line)