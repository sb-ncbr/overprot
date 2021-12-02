'''
Extract C-alpha atoms of non-hetatm residues.
'''

from pathlib import Path
import argparse

from libs import lib_pymol

#  PARSE ARGUMENTS  ################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Input structure (mmCIF file)', type=Path)
parser.add_argument('output', help='Filename for output structure', type=Path)
args = parser.parse_args()

lib_pymol.extract_alpha_trace(args.input, args.output)