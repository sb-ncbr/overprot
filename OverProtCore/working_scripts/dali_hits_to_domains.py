# Creates intersection of input domain ranges with observed residues from PDBe API. Input and output is in JSON format { pdb: [[domain_name, chain, range]] }.

import sys
import re
import requests
import json
import argparse
import math
from typing import List, Tuple, Dict
from collections import defaultdict

#  PARSE ARGUMENTS  ################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('dali_file', help='Column-based file with result of DALI structural search', type=str)
parser.add_argument('--max_loop', help='Maximum number of contiguous unmatched residues to be included in the domain (default: infinity)', type=int, default=math.inf)
args = parser.parse_args()

dali_file = args.dali_file
max_loop = args.max_loop

#  FUNCTIONS  ################################################################################

def include_loops(ranges: List[Tuple[int,int]], max_loop=math.inf) -> List[Tuple[int,int]]:
    """Include loops in domain ranges, e.g. '1:100,150:200' -> [(1,100), (150,200)], except for the loops longer than max_loop residues."""
    result = []
    for rang in ranges:
        if len(result) > 0 and rang[0] - result[-1][1] - 1 <= max_loop:  # short loop -> include loop
            result[-1] = (result[-1][0], rang[1])
        else:  # long loop (or the first range) -> exclude loop
            result.append(rang)
    return result

def format_ranges(ranges: List[Tuple[int,int]]) -> str:
    """Format domain ranges, e.g. [(1,100), (150,200)] -> '1:100,150:200'"""
    return ','.join(f'{r[0]}:{r[1]}' for r in ranges)

#  MAIN  ###############################################################################

re_struct_equiv = re.compile('^ *(\d+): *(\w{4})-(\w+) +(\w{4})-(\w+).*\( *(\w+) +(\d+) *- *(\w+) +(\d+) *<=> *(\w+) +(\d+) *- *(\w+) +(\d+) *\)$')

matches = defaultdict(list)  # {i: [(pdb, chain, start_resi, end_resi)...]}

with open(dali_file) as f:
    for line in f:
        if not line.startswith('#'):
            line = line.rstrip()
            match = re_struct_equiv.match(line)
            if match is not None:
                fields = [match.group(i) for i in range(1, 14)]
                i_match, q_pdb, q_chain, s_pdb, s_chain, q_start_resn, q_start_resi, q_end_resn, q_end_resi, s_start_resn, s_start_resi, s_end_resn, s_end_resi = fields
                matches[i_match].append((s_pdb, s_chain, s_start_resi, s_end_resi))

domain_names = [ranges[0][0] + ranges[0][1] for ranges in matches.values()]

if len(domain_names) != len(set(domain_names)):
    raise NotImplementedError('More than one hit in one PDB chain')

pdb2domains = defaultdict(list)

for pdbs_chains_starts_ends in matches.values():
    pdb = pdbs_chains_starts_ends[0][0]
    chain = pdbs_chains_starts_ends[0][1]
    ranges = [(int(start), int(end)) for pdb, chain, start, end in pdbs_chains_starts_ends]
    ranges = include_loops(ranges, max_loop=max_loop)
    # start = min(int(rang[2]) for rang in ranges)
    # end = max(int(rang[3]) for rang in ranges)
    domain_name = pdb + chain
    pdb2domains[pdb].append((domain_name, chain, format_ranges(ranges)))

pdb2domains = {pdb: sorted(doms) for pdb, doms in sorted(pdb2domains.items())}

print(json.dumps(pdb2domains, indent=4))

print(f'Output: {sum(len(doms) for doms in pdb2domains.values())} domains in {len(pdb2domains)} PDB entries', file=sys.stderr)


# # No:  Chain   Z    rmsd lali nres  %id PDB  Description
#    1:  6lzg-B 35.1  0.0  195   195  100   MOLECULE: ANGIOTENSIN-CONVERTING ENZYME 2;          
# # Structural equivalences
#    1: 6lzg-B 6lzg-B     1 - 195 <=>    1 - 195   (THR  333  - PRO  527  <=> THR  333  - PRO  527 )  
