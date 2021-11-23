# Creates intersection of input domain ranges with observed residues from PDBe API. Input and output is in JSON format { pdb: [[domain_name, chain, range]] }.

import sys
import requests
import json
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict

#  PARSE ARGUMENTS  ################################################################################

DEFAULT_API_URL = 'https://www.ebi.ac.uk/pdbe/api/pdb/entry/polymer_coverage'

parser = argparse.ArgumentParser()
parser.add_argument('input_domains', help='JSON file with input domains in format { pdb: [[domain_name, chain, range]] }', type=str)
parser.add_argument('--source', help='URL with PDBeAPI server for Observed residues (default = ' + DEFAULT_API_URL + ')', default=DEFAULT_API_URL)
parser.add_argument('--min_residues', help='Remove domains with less than this number of residues (default: 0)', type=int, default=0)
args = parser.parse_args()

input_domains_file = args.input_domains
api_url = args.source
min_residues = args.min_residues

#  FUNCTIONS  ################################################################################

def parse_ranges(ranges: str) -> List[Tuple[int,int]]:
    """Parse domain ranges, e.g. '1:100,150:200' -> [(1,100), (150,200)]"""
    result = []
    for rang in ranges.split(','):
        start, end = rang.split(':')
        start = int(start) if start != '' else float('-inf')
        end = int(end) if end != '' else float('+inf')
        result.append((start, end))
    return result

def format_ranges(ranges: List[Tuple[int,int]]) -> str:
    """Format domain ranges, e.g. [(1,100), (150,200)] -> '1:100,150:200'"""
    return ','.join(f'{r[0]}:{r[1]}' for r in ranges)

def get_observed_residues(pdb: str, source_api_url: str) -> Dict[str, List[Tuple[int,int]]]:
    url = f'{source_api_url}/{pdb}'
    response = requests.get(url)
    info = json.loads(response.text)
    molecules = info[pdb]['molecules']
    result = {}
    for molecule in molecules:
        for chain in molecule['chains']:
            chain_id = chain['struct_asym_id']
            ranges_in_chain = []
            for rang in chain['observed']:
                start = rang['start']['residue_number']
                end = rang['end']['residue_number']
                ranges_in_chain.append((start, end))
            ranges_in_chain.sort()
            assert chain_id not in result
            result[chain_id] = ranges_in_chain
    return result

def consolidate_ranges(ranges: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Return equivalent set of ranges, which is without overlaps and sorted, e.g. [(120,155), (1,100), (150,200)] -> [(1,100), (120,200)]"""
    stack = sorted(ranges, reverse=True)
    result = []
    while len(stack) >= 2:
        head = stack[-1]
        neck = stack[-2]
        if neck[0] <= head[1] + 1: 
            union = (head[0], max(head[1], neck[1]))
            stack.pop()
            stack.pop()
            stack.append(union)
        else:
            stack.pop()
            result.append(head)
    result.extend(stack)
    return result

def ranges_intersection(ranges1: List[Tuple[int,int]], ranges2: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Return the intersection of two sets of ranges, e.g. [(1,100), (150,200)] + [(50,180)] -> [(50,100), (150,180)]"""
    stack1 = sorted(consolidate_ranges(ranges1), reverse=True)
    stack2 = sorted(consolidate_ranges(ranges2), reverse=True)
    result = []
    while len(stack1) > 0 and len(stack2) > 0:
        head1 = stack1[-1]
        head2 = stack2[-1]
        overlap_start = max(head1[0], head2[0])
        overlap_end = min(head1[1], head2[1])
        if overlap_start <= overlap_end:
            result.append((overlap_start, overlap_end))
        if head1[1] < head2[1]:
            stack1.pop()
        else:
            stack2.pop()
    return result

def ranges_residue_count(ranges: List[Tuple[int,int]]) -> int:
    """Return total number of residues in domain ranges, e.g. [(1,100), (150,200)] -> 151"""
    return sum(r[1] - r[0] + 1 for r in ranges)


#  MAIN  ###############################################################################

with open(input_domains_file) as f:
    pdb2domains = json.load(f)

result = defaultdict(list)

print(f'Input: {sum(len(doms) for doms in pdb2domains.values())} domains in {len(pdb2domains)} PDB entries', file=sys.stderr)

for pdb, domains in pdb2domains.items():
    observed = get_observed_residues(pdb, api_url)
    # print(pdb, observed)
    for domain_name, chain, ranges in domains:
        domain_ranges = parse_ranges(ranges)
        observed_ranges = observed.get(chain, [])
        overlap = ranges_intersection(domain_ranges, observed_ranges)
        print(ranges_residue_count(overlap), pdb, chain, format_ranges(domain_ranges), format_ranges(observed_ranges), format_ranges(overlap), file=sys.stderr)
        if ranges_residue_count(overlap) >= min_residues:
            new_domain = (domain_name, chain, format_ranges(overlap))
            result[pdb].append(new_domain)
            # print(f'    {chain}: {domain_ranges}  {overlap}')
        else:
            pass
            # print(f'Removing  domain: {pdb} {chain} {format_ranges(domain_ranges)} (observed: {format_ranges(observed_ranges)})', file=sys.stderr)

json.dump(result, sys.stdout, indent=4)

print(f'Output: {sum(len(doms) for doms in result.values())} domains in {len(result)} PDB entries', file=sys.stderr)

# url = api_url + '/' + accession
# sys.stderr.write('Downloading ' + url + '\n')
# response = requests.get(url)
# if response.ok:
#     results = json.loads(response.text).get(accession, {}).get('PDB', {})
# else: 
#     sys.stderr.write('HTTP request failed, status code ' + str(response.status_code) + '.\n')
#     exit(1)

# output = {}
# for pdb, entry in results.items():
#     if isinstance(entry, list):
#         output[pdb] = get_domains_multisegment(entry, pdb)
#     elif isinstance(entry, dict) and 'mappings' in entry:
#         output[pdb] = get_domains_multisegment(entry['mappings'], pdb)

# n_pdbs = len(output)
# n_domains = sum( len(doms) for pdb, doms in output.items() )

# sys.stderr.write(f'Found {n_domains} domains in {n_pdbs} PDB entries.\n')

# print(json.dumps(output, sort_keys=True, indent=4))