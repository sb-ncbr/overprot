'''
Select a random sample from a set of domains.

Example usage:
    python3  -m overprot.select_random_domains  --help
'''

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from typing import Optional

from .libs import lib_domains
from .libs import lib
from .libs.lib_cli import cli_command, run_cli_command

#  CONSTANTS  ################################################################################

RANDOMIZE_DOMAINS_WITHIN_PDB = False

#  MAIN  #####################################################################################

@cli_command(parsers={'size': lib.int_or_all})
def main(domain_file: Path, 
         size: Optional[int] = None, or_all: bool = False,
         unique_pdb: bool = False, unique_uniprot: bool = False) -> Optional[int]:
    '''Select a random sample from a set of domains.
    @param  `domain_file`     JSON file with format {pdb: [[domain_name, chain, range]]}.
    @param  `size`            Size of the selected sample (integer or 'all'). [default: "all"]
    @param  `or_all`          Select all domains if the number of domains is smaller than the requested size (otherwise would raise an error).
    @param  `unique_pdb`      Take only the first domain listed for each PDB code.
    @param  `unique_uniprot`  Take only the first domain listed for each UniProtID (input must be sorted by UniProtID).
    '''
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
    sample_size = N if size is None else size
    
    if sample_size > N:
        if or_all:
            sample_size = N
        else:
            raise Exception(f"Required sample size ({sample_size}) is larger than the total number of domains ({N}). You can use '--size all' to select all domains or '--size N --or_all' to select at most N domains.")
    
    indices = sorted(np.random.choice(N, size=sample_size, replace=False))
    sample = [domains[i] for i in indices]

    # Output
    lib.dump_json(sample, sys.stdout)

    str_all = ' (all)' if sample_size==N else ''
    str_unique_pdb = ' unique-PDB' if unique_pdb else ''
    print(f'Selected {sample_size}{str_all} out of {N}{str_unique_pdb} domains', file=sys.stderr)

    return None

if __name__ == '__main__':
    run_cli_command(main)
