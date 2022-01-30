'''
Download file with CATH example domains for all families and print as CSV.

Example usage:
    python3  -m overprot.get_cath_example_domains  --help
'''

import requests
import json
from pathlib import Path
from typing import Optional

from .libs.lib_io import RedirectIO
from .libs.lib_cli import cli_command, run_cli_command

#  CONSTANTS  ################################################################################

DEFAULT_URL = 'http://cathdb.info/version/v4_3_0/api/rest/cathtree/from_cath_id_to_depth/root/4?content-type=application/json'

#  FUNCTIONS  ################################################################################

def process_node(js: dict) -> None:
    node_id = js.get('cath_id', js['cath_id_padded'])
    example = js.get('example_domain_id')
    children = js.get('children', [])
    depth = int(js['cath_id_depth'])
    assert depth >= 0
    assert depth <= 4
    if depth > 0:
        assert example is not None
        print(node_id, example, sep=';')
    for child in children:
        process_node(child)

#  MAIN  #####################################################################################

@cli_command()
def main(url: Optional[str] = DEFAULT_URL, input_json: Optional[Path] = None, output: Optional[Path] = None) -> Optional[int]:
    '''Download file with CATH example domains for all families and print as CSV.
    @param  `url`         Address of the downloaded file.
    @param  `input_json`  Do not download but read from this local file.
    @param  `output`      Output file.
    '''
    if input_json is None:
        response = requests.get(url)
        assert response.status_code == 200
        js = json.loads(response.text)
    else: 
        with open(input_json) as f:
            js = json.load(f)
    with RedirectIO(stdout=output):
        print('cath_node;example_domain')
        process_node(js)
    

if __name__ == '__main__':
    run_cli_command(main)
