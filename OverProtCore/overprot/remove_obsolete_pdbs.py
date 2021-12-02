'''
This Python3 script checks suspicious PDB entry whether they are obsolete. If yes it removes them from sample_json (printed to output), otherwise it raises an error.

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import argparse
from pathlib import Path
import json
import requests
from typing import Dict, Any, Optional

from .libs import lib_domains
from .libs import lib

#  CONSTANTS  ################################################################################

PDBE_API_RELEASE_STATUS = 'https://www.ebi.ac.uk/pdbe/api/pdb/entry/status/{pdb}'

#  FUNCTIONS  ################################################################################

class MissingPDBError(Exception):
    pass

def is_obsolete(pdb: str) -> bool:
    response = requests.get(PDBE_API_RELEASE_STATUS.format(pdb=pdb)).content
    json_response = json.loads(response)
    if pdb not in json_response:
        raise MissingPDBError(f'Missing PDB entry: {pdb}')
    assert len(json_response[pdb]) == 1
    return json_response[pdb][0]['status_code'].upper() == 'OBS'

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample_json', help='Input file with domains', type=Path)
    parser.add_argument('missing_list', help='File with the list of missing (possibly obsolete) PDB structures', type=Path)
    parser.add_argument('output_sample_json', help='Output file with non-obsolete domains', type=Path)
    parser.add_argument('output_missing_list', help='Filename for the list of missing non-obsolete PDB structures', type=Path)
    # TODO add command line arguments
    args = parser.parse_args()
    return vars(args)


def main(sample_json: Path, missing_list: Path, 
         output_sample_json: Path, output_missing_list: Path) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    domains = lib_domains.load_domain_list(sample_json)
    obsolete = []
    still_missing = []
    with open(missing_list) as r:
        missing = r.read().split()
    for pdb in missing:
        if is_obsolete(pdb):
            obsolete.append(pdb)
        else:
            still_missing.append(pdb)
    with open(output_missing_list, 'w') as w:
        for pdb in still_missing:
            w.write(pdb)
            w.write('\n')
    filtered_domains = [dom for dom in domains if dom.pdb not in obsolete]
    lib.dump_json(filtered_domains, output_sample_json)
    return 0 if len(still_missing) == 0 else 1


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)