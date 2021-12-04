'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import argparse
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional

#  CONSTANTS  ################################################################################


URL = 'http://cathdb.info/version/v4_3_0/api/rest/cathtree/from_cath_id_to_depth/root/4?content-type=application/json'

#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_json', help=f'Input JSON (default: download from {URL})', type=Path)
    args = parser.parse_args()
    return vars(args)


def main(input_json: Optional[Path] = None) -> Optional[int]:
    # TODO add parameters
    '''Foo'''
    # TODO add docstring
    if input_json is None:
        response = requests.get(URL)
        assert response.status_code == 200
        # print(response.text)
        js = json.loads(response.text)
    else: 
        with open(input_json) as f:
            js = json.load(f)
    process_node(js)
    
def process_node(js: dict) -> None:
    node_id = js.get('cath_id', js['cath_id_padded'])
    example = js.get('example_domain_id')
    children = js.get('children', [])
    depth = int(js['cath_id_depth'])
    assert depth >= 0
    assert depth <= 4
    if depth > 0:
        assert example is not None
        print(node_id, example)
    # print('    '*depth, repr(node_id), example, len(children))
    for child in children:
        process_node(child)


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)