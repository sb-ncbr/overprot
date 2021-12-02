'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from .libs import lib_domains
from .libs import lib_pymol

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_file', help='CIF file with target structure', type=Path)
    parser.add_argument('sample_file', help=f'JSON file with list of domains to be aligned', type=Path)
    parser.add_argument('in_directory', help='Directory with input structures (named {domain_name}.cif)', type=Path)
    parser.add_argument('out_directory', help='Directory for output structures (named {domain_name}.cif)', type=Path)
    parser.add_argument('--progress_bar', help='Show progress bar', action='store_true')
    args = parser.parse_args()
    return vars(args)


def main(target_file: Path, sample_file: Path, in_directory: Path, out_directory: Path, progress_bar: bool = False) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    domains = lib_domains.load_domain_list(sample_file)
    mobiles = [dom.name for dom in domains]
    out_directory.mkdir(parents=True, exist_ok=True)
    mobile_files = [in_directory/f'{mobile}.cif' for mobile in mobiles]
    result_files = [out_directory/f'{mobile}.cif' for mobile in mobiles]
    result_ttt_files = [out_directory/f'{mobile}-ttt.csv' for mobile in mobiles]
    lib_pymol.cealign_many(target_file, mobile_files, result_files, ttt_files=result_ttt_files, fallback_to_dumb_align=True, show_progress_bar=progress_bar)
    return None


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)