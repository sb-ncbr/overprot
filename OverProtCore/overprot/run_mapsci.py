'''
This Python3 script prepares MAPSCI input file and runs MAPSCI (multiple structural alignment program)

Example usage:
    python3 run_mapsci.py domains.json input_dir/ output_dir/ --mapsci ./mapsci --init center --n_max 100 --keep_rotated
'''
# TODO add description and example usage in docstring

import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Final

from .libs import lib
from .libs import lib_sh
from .libs import lib_domains
from .libs.lib_dependencies import MAPSCI_EXE

#  CONSTANTS  ################################################################################

DEFAULT_INIT: Final = 'center'

N_MAX_ALL = -1

MAPSCI_INPUT_FILE_NAME = 'mapsci_input.txt'
MAPSCI_STDOUT = 'stdout.txt'
MAPSCI_STDERR = 'stderr.txt'
MAPSCI_CONSENSUS_FILE = 'consensus.pdb'
STRUCTURE_EXT = '.pdb'
ROTATED_EXT = '.pdb.rot'

#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='File with the list of protein domains in format [[pdb, domain_name, chain, range]]', type=Path)
    parser.add_argument('input_dir', help='Directory with input PDB files, named <domain_name>.pdb', type=Path)
    parser.add_argument('output_dir', help='Directory for output', type=Path)
    parser.add_argument('--init', help=f'Initial consensus selection method (default: {DEFAULT_INIT}, fastest: median, see MAPSCI documentation)', type=str, choices=['center', 'minmax', 'median'], default='center')
    parser.add_argument('--n_max', help=f'Maximum number of input domains. If there are more input domains, then N_MAX domains are selected randomly. Default: {N_MAX_ALL} (always take all).', type=int, default=N_MAX_ALL)
    parser.add_argument('--keep_rotated', help=f'Do not delete the rotated structure files ({ROTATED_EXT}) produced by MAPSCII', action='store_true')
    args = parser.parse_args()
    return vars(args)


def main(input_file: Path, input_dir: Path, output_dir: Path, 
         init: Literal['center', 'minmax', 'median'] = DEFAULT_INIT, 
         n_max: int = N_MAX_ALL, keep_rotated: bool = False) -> Optional[int]:
    '''Prepare MAPSCI input file and run MAPSCI'''
    # TODO add docstring

    # Convert to absolute paths (important when calling MAPSCI)
    input_file = input_file.resolve()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    # Read input domain list
    domains = lib_domains.load_domain_list(input_file)

    # Select random subset, if too many domains
    if n_max != N_MAX_ALL and len(domains) > n_max:
        print(f'Selecting {n_max} out of {len(domains)} domains')
        selected_indices = sorted(lib.consistent_pseudorandom_choice((dom.name for dom in domains), n_max))
        # selected_indices = sorted(numpy.random.choice(len(domains), n_max, replace=False))
        domains = [domains[i] for i in selected_indices]
    else:
        print(f'Taking all {len(domains)} domains')

    # Prepare input file for MAPSCI
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/MAPSCI_INPUT_FILE_NAME, 'w') as w:
        for domain in domains:
            w.write(domain.name + STRUCTURE_EXT + '\n')

    # Run MAPSCI
    with open(output_dir/MAPSCI_STDOUT, 'w') as stdout_writer:
        with open(output_dir/MAPSCI_STDERR, 'w') as stderr_writer:
            subprocess.run([MAPSCI_EXE, MAPSCI_INPUT_FILE_NAME, init, '-p', str(input_dir)], cwd=str(output_dir), stdout=stdout_writer, stderr=stderr_writer)


    ok = (output_dir/MAPSCI_CONSENSUS_FILE).is_file()
    if ok:
        print('MAPSCI OK')
    else:
        print('MAPSCI FAILED')
        try:
            with open(output_dir/MAPSCI_STDERR) as r:
                error_message = r.readlines()[-1]
        except (OSError, IndexError):
            error_message = ''
        raise Exception(f'MAPSCI failed: {error_message}')

    # Delete rotated structure files
    if not keep_rotated:
        for domain in domains:
            lib_sh.rm(output_dir / (domain.name+ROTATED_EXT), ignore_errors=True)
    
    return None


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)
