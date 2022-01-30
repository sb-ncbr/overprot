'''
Prepare MAPSCI input file and run MAPSCI.

Example usage:
    python3  -m overprot.run_mapsci  domains.json  input_dir/  output_dir/  --init center  --n_max 100  --keep_rotated
'''

import subprocess
from pathlib import Path
from typing import Literal, Final

from .libs import lib
from .libs import lib_sh
from .libs import lib_domains
from .libs.lib_dependencies import MAPSCI_EXE
from .libs.lib_cli import cli_command, run_cli_command


DEFAULT_INIT: Final = 'center'

N_MAX_ALL = -1

MAPSCI_INPUT_FILE_NAME = 'mapsci_input.txt'
MAPSCI_STDOUT = 'stdout.txt'
MAPSCI_STDERR = 'stderr.txt'
MAPSCI_CONSENSUS_FILE = 'consensus.pdb'
STRUCTURE_EXT = '.pdb'
ROTATED_EXT = '.pdb.rot'


@cli_command()
def main(input_file: Path, input_dir: Path, output_dir: Path, 
         init: Literal['center', 'minmax', 'median'] = DEFAULT_INIT, 
         n_max: int = N_MAX_ALL, keep_rotated: bool = False) -> None:
    '''Prepare MAPSCI input file and run MAPSCI.
    @param  `input_file`    File with the list of protein domains in format [[pdb, domain_name, chain, range]].
    @param  `input_dir`     Directory with input PDB files, named {domain_name}.pdb.
    @param  `output_dir`    Directory for output.
    @param  `init`          Initial consensus selection method (fastest: median, see MAPSCI documentation).
    @param  `n_max`         Maximum number of input domains. If there are more input domains, then `n_max` domains are selected quasi-randomly (default: always take all).
    @param  `keep_rotated`  Do not delete the rotated structure files produced by MAPSCII.
    '''
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
    

if __name__ == '__main__':
    run_cli_command(main)
