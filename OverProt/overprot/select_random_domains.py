'''
Selects a random sample from a set of domains.

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import json
import os
from os import path
import sys
import argparse
import numpy as np
from typing import Dict, Any, Optional, Union, Literal

from .libs import lib_domains
from .libs.lib import FilePath

#  CONSTANTS  ################################################################################

RANDOMIZE_DOMAINS_WITHIN_PDB = False

#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain_file', help='JSON file with format {pdb: [[domain_name, chain, range]]}', type=str)
    # parser.add_argument('directory', help='Directory to save samples.json and downloaded PDB files', type=str)
    parser.add_argument('--size', help="Size of the selected sample (integer or 'all')", type=str, default='all')
    parser.add_argument('--or_all', help='Select all domains if the number of domains is smaller than the requested size (otherwise would raise an error)', action='store_true')
    parser.add_argument('--unique_pdb', help='Take only the first domain listed for each PDB code', action='store_true')
    parser.add_argument('--unique_uniprot', help='Take only the first domain listed for each UniProtID (input must be sorted by UniProtID)', action='store_true')
    # parser.add_argument('--download', help='Fetch PDB files for selected domains (requires pymol module)', action='store_true')
    # parser.add_argument('--copy_from', help='Copy PDB files for selected domains from specified directory', type=str, default=None)
    # parser.add_argument('--cif', help='Use CIF format instead of PDB.', action='store_true')
    args = parser.parse_args()
    return vars(args)
    # TODO add mutually exclusive group


def main(domain_file: Union[FilePath, str], 
         size: Union[int, str, None] = 'all', or_all: bool = False,
         unique_pdb: bool = False, unique_uniprot: bool = False) -> Optional[int]:
    '''Select a random sample from a set of domains.'''
    
    # Parse input file
    domains_by_pdb = lib_domains.load_domain_list_by_pdb(domain_file)

    if unique_uniprot:
        raise NotImplementedError
        # TODO implement using e.g. ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/pdb_chain_cath_uniprot.csv, not PDBeAPI (request per entry)
        # for uni, pdbs_domains in pdb_dict.items(): # expects different format of input
        #     pdb = list(pdbs_domains)[0]
        #     domain, chain, rang = pdbs_domains[pdb][0]
        #     domains.append((pdb, domain, chain, rang))
    elif unique_pdb:
        if RANDOMIZE_DOMAINS_WITHIN_PDB:
            domains = [np.random.choice(doms) for doms in domains_by_pdb.values()]
        else:
            domains = [doms[0] for doms in domains_by_pdb.values()]
    else:
        domains = [dom for doms in domains_by_pdb.values() for dom in doms]
    
    # Select sample
    N = len(domains)
    sample_size = N if (size == 'all' or size is None) else int(size)
    
    if sample_size > N:
        if or_all:
            sample_size = N
        else:
            raise Exception(f"Required sample size ({sample_size}) is larger than the total number of domains ({N}). You can use '--size all' to select all domains or '--size N --or_all' to select at most N domains.")
    
    indices = sorted(np.random.choice(N, size=sample_size, replace=False))
    sample = [domains[i] for i in indices]

    # Output
    json.dump(sample, sys.stdout, indent=4)

    str_all = ' (all)' if sample_size==N else ''
    str_unique_pdb = ' unique-PDB' if unique_pdb else ''
    print(f'Selected {sample_size}{str_all} out of {N}{str_unique_pdb} domains', file=sys.stderr)

    return None

if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)



# Selects a random sample from a set of domains.

################################################################################
