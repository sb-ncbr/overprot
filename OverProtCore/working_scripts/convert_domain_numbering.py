# Creates intersection of input domain ranges with observed residues from PDBe API. Input and output is in JSON format { pdb: [[domain_name, chain, range]] }.

import sys
from pathlib import Path
import requests
import json
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict

#  PARSE ARGUMENTS  ################################################################################

DEFAULT_RESIDUE_LISTING_API_URL = 'https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/'
DEFAULT_STATUS_API_URL = 'https://www.ebi.ac.uk/pdbe/api/pdb/entry/status/'

parser = argparse.ArgumentParser()
parser.add_argument('conversion', help='Conversion direction', type=str, choices=['auth2label', 'label2auth'])
parser.add_argument('input_domains', help='JSON file with input domains in format { pdb: [[domain_name, chain, range]] }', type=Path)
parser.add_argument('--source', help='URL with PDBeAPI server for List of residues (default = ' + DEFAULT_RESIDUE_LISTING_API_URL + ')', default=DEFAULT_RESIDUE_LISTING_API_URL)
parser.add_argument('--ignore_obsolete', help='Ignore obsolete entries without raising error', action='store_true')
parser.add_argument('--insertion_warnings', help='Print warnings when residues with insertion codes are found', action='store_true')
args = parser.parse_args()

input_domains_file = args.input_domains
conversion = args.conversion
residue_listing_api_url = args.source
ignore_obsolete = args.ignore_obsolete
insertion_warnings = args.insertion_warnings

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

def get_conversion_table(pdb: str, conversion='auth2label', residue_listing_api_url: str = DEFAULT_RESIDUE_LISTING_API_URL) -> Dict[str, List[Tuple[int,int,int,int]]]:
    assert conversion == 'auth2label' or conversion == 'label2auth', "conversion must be 'auth2label' or 'label2auth'"
    url = f'{residue_listing_api_url}/{pdb}'
    response = requests.get(url)
    info = json.loads(response.text)
    molecules = info[pdb]['molecules']
    table = {}
    for molecule in molecules:
        for chain in molecule['chains']:
            label_chain = chain['struct_asym_id']
            auth_chain = chain['chain_id']
            for residue in chain['residues']:
                label_resi = int(residue['residue_number'])
                auth_resi = int(residue['author_residue_number'])
                auth_ins_code = residue['author_insertion_code']
                if auth_ins_code != '':
                    if insertion_warnings:
                        print(f'WARNING: Ignoring residue with insertion code: {pdb} {auth_chain} {auth_resi}{auth_ins_code}', file=sys.stderr)
                else:
                    if conversion == 'auth2label':
                        table[(auth_chain, auth_resi)] = (label_chain, label_resi)
                    else:
                        table[(label_chain, label_resi)] = (auth_chain, auth_resi)
    return table

def is_obsolete(pdb: str, status_api_url: str = DEFAULT_STATUS_API_URL):
    url = f'{status_api_url}/{pdb}'
    response = requests.get(url)
    info = json.loads(response.text)
    status = info[pdb][0]['status_code']
    return status == 'OBS'

#  MAIN  ###############################################################################

with open(input_domains_file) as f:
    pdb2domains = json.load(f)

result = defaultdict(list)

print(f'Input: {sum(len(doms) for doms in pdb2domains.values())} domains in {len(pdb2domains)} PDB entries', file=sys.stderr)

for pdb, domains in pdb2domains.items():
    try:
        table = get_conversion_table(pdb, conversion, residue_listing_api_url=residue_listing_api_url)
        for domain_name, chain, ranges in domains:
            ranges = parse_ranges(ranges)
            new_ranges = []
            for start, end in ranges:
                new_chain, new_start = table[(chain, start)]
                new_chain, new_end = table[(chain, end)]
                new_ranges.append((new_start, new_end))
            new_domain = (domain_name, new_chain, format_ranges(new_ranges))
            result[pdb].append(new_domain)
    except KeyError as ex:
        if ignore_obsolete and is_obsolete(pdb):
            continue
        else:
            raise ex
json.dump(result, sys.stdout, indent=4)
