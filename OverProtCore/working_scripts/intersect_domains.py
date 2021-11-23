# Creates intersection of input domain ranges from two files. Input and output is in JSON format { pdb: [[domain_name, chain, range]] }.

import sys
import requests
import json
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict

#  PARSE ARGUMENTS  ################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('input_domains_1', help='JSON file with input domains in format { pdb: [[domain_name, chain, range]] }', type=str)
parser.add_argument('input_domains_2', help='JSON file with input domains in format { pdb: [[domain_name, chain, range]] }', type=str)
args = parser.parse_args()

input_domains_file_1 = args.input_domains_1
input_domains_file_2 = args.input_domains_2

#  FUNCTIONS  ################################################################################

def parse_ranges(ranges: str) -> List[Tuple[int,int]]:
    """Parse domain ranges, e.g. '1:100,150:200' -> [(1,100), (150,200)]"""
    result = []
    for rang in ranges.split(','):
        start, end = rang.split(':')
        result.append((int(start), int(end)))
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

def get_ranges_in_chain(domains: List[Tuple[str,str,str]], chain: str) -> List[Tuple[int,int]]:
    result = []
    for domain_name, chain, ranges in domains:
        result.extend(parse_ranges(ranges))
    result = consolidate_ranges(result)
    return result

#  MAIN  ###############################################################################

with open(input_domains_file_1) as f:
    pdb2domains1 = json.load(f)

with open(input_domains_file_2) as f:
    pdb2domains2 = json.load(f)

result = defaultdict(list)

print(f'Input 1: {sum(len(doms) for doms in pdb2domains1.values())} domains in {len(pdb2domains1)} PDB entries', file=sys.stderr)
print(f'Input 2: {sum(len(doms) for doms in pdb2domains2.values())} domains in {len(pdb2domains2)} PDB entries', file=sys.stderr)

pdbs = set(pdb2domains1.keys()) & set(pdb2domains2.keys())

result = defaultdict(list)

for pdb in pdbs:
    domains1 = pdb2domains1[pdb]
    domains2 = pdb2domains2[pdb]
    for domain_name1, chain1, ranges1 in domains1:
        ranges1 = parse_ranges(ranges1)
        ranges2 = get_ranges_in_chain(domains2, chain1)
        overlap = ranges_intersection(ranges1, ranges2)
        if ranges_residue_count(overlap) > 0:
            new_domain = (domain_name1, chain1, format_ranges(overlap))
            result[pdb].append(new_domain)

json.dump(result, sys.stdout, indent=4)

print(f'Output: {sum(len(doms) for doms in result.values())} domains in {len(result)} PDB entries', file=sys.stderr)
