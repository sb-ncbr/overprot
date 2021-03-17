'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import argparse
import json
from typing import Dict, Any, Optional, Union

from .libs.lib import FilePath

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', help='Input directory with families.txt etc. (passed to overprot_multifamily.py as directory).', type=str)
    parser.add_argument('output_dir', help='Output directory.', type=str)
    args = parser.parse_args()
    return vars(args)


def main(input_dir: Union[FilePath, str], output_dir: Union[FilePath, str]) -> Optional[int]:
    '''Foo'''
    # TODO add docstring

    indir = FilePath(input_dir)
    outdir = FilePath(output_dir)
    with indir.sub('families.txt').open() as r:
        families = r.read().split()

    if outdir.isdir():
        outdir.rm(recursive=True)
    outdir.mkdir()
    for family in families:
        inp = indir.sub('families', family, 'results', 'diagram.json')
        out = outdir.sub(f'diagram-{family}.json')
        if inp.isfile():
            inp.cp(out)
        else:
            with out.open('w') as w:
                js_error = {'error': f"Failed to generate consensus for '{family}'"}
                json.dump(js_error, w)
    return 0


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)