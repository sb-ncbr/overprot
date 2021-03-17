'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import sys
import argparse
import json
from typing import Dict, Any, Optional, Union

from .libs import lib_domains
from .libs.lib import FilePath

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='JSON file with domain list', type=str)
    # TODO add command line arguments
    args = parser.parse_args()
    return vars(args)


def main(input_file: Union[FilePath, str]) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    domains = lib_domains.load_domain_list(input_file)
    simplified = [(dom.pdb, dom.name, dom.chain, dom.ranges) for dom in domains]
    json.dump(simplified, sys.stdout, indent=4)
    return None


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)