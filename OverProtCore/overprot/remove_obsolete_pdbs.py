'''
Check missing PDB entries, if they are not obsolete.  

Example usage:
    python3  -m overprot.remove_obsolete_pdbs  --help
'''

from pathlib import Path
import json
import requests

from .libs import lib_domains
from .libs import lib
from .libs.lib_cli import cli_command, run_cli_command

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

@cli_command()
def main(sample_json: Path, missing_list: Path, 
         output_sample_json: Path, output_missing_list: Path) -> int:
    '''Check missing PDB entries, whether they are not obsolete. 
    Return 1 if any non-obsolete entries are still missing, 0 otherwise.
    @params  `sample_json`          Input file with all domains.
    @params  `missing_list`         Input file with the list of missing (possibly obsolete) PDB entries.
    @params  `output_sample_json`   Output file with non-obsolete domains.
    @params  `output_missing_list`  Output file with the list of missing non-obsolete PDB entries.
    '''
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
    run_cli_command(main)
