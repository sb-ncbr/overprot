import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, Any, Optional, Union, List, Literal

from overprot import format_domains

#  FUNCTIONS  ################################################################################

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', help='Data directory from overprot_multifamily.py', type=str)
    args = parser.parse_args()
    return vars(args)

def main(data_dir: str) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    family_dirs = [d for d in Path(data_dir, 'families').iterdir() if d.is_dir()]
    print(len(family_dirs), 'families')

    for d in family_dirs:
        lists = d/'lists'
        lists.mkdir(exist_ok=True)
        format_domains.main(d/'family.json', d/'sample.json', 
            pdbs_json=lists/'pdbs.json', pdbs_csv=lists/'pdbs.csv', pdbs_html=lists/'pdbs.html',
            domains_json=lists/'domains.json', domains_csv=lists/'domains.csv', domains_html=lists/'domains.html',
            sample_json=lists/'sample.json', sample_csv=lists/'sample.csv', sample_html=lists/'sample.html')
        shutil.copy(d/'family_info.txt', lists/'family_info.txt')
        if Path(d, 'family.json').exists():
            shutil.copy(d/'family.json', lists/'family.json')
        else:
            with open(lists/'family.json', 'w') as w:
                json.dump({}, w)

    collected = Path(data_dir, 'collected_results', 'families')
    collected.mkdir(exist_ok=True)
    for d in family_dirs:
        family = d.name
        shutil.copytree(d/'lists', collected/family)

    return None

def _main():
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)

if __name__ == '__main__':
    _main()

# #############################################################################################

