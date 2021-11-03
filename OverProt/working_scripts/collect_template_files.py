'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import sys
import argparse
from pathlib import Path
import uuid
import shutil
from typing import Dict, Any, Optional

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('families_dir', help='Directory with per-family subdirs from OverProt', type=Path)
    parser.add_argument('output_file', help='Filename for the output archive (.zip or .tar.gz) or directory', type=Path)
    args = parser.parse_args()
    return vars(args)


def main(families_dir: Path, output_file: Path) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    if str(output_file).endswith('.zip'):
        output_basename = str(output_file)[:-len('.zip')]
        archive_format = 'zip'
    elif str(output_file).endswith('.tar.gz'):
        output_basename = str(output_file)[:-len('.tar.gz')]
        archive_format = 'gztar'
    else:
        output_basename = str(output_file)
        archive_format = ''
    output_dir = Path(f'{output_basename}.{uuid.uuid4().hex}.tmp') if archive_format else Path(output_basename)
    output_dir.mkdir(parents=True)
    for famdir in sorted(families_dir.iterdir()):
        family = famdir.name
        try:
            shutil.copy(famdir / 'results' / 'consensus.cif', output_dir / f'consensus-{family}.cif')
            shutil.copy(famdir / 'results' / 'consensus.sses.json', output_dir / f'consensus-{family}.sses.json')
        except FileNotFoundError:
            print(f'WARNING: missing files for family {family}', file=sys.stderr)
    if archive_format:
        shutil.make_archive(output_basename, archive_format, output_dir)
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)