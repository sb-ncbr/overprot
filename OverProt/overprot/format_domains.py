import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Literal

#  FUNCTIONS  ################################################################################

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_family_json', help='Input family.json', type=str)
    parser.add_argument('input_sample_json', help='Input sample.json', type=str)
    parser.add_argument('--pdbs_html', help='Output pdbs.html', type=str)
    parser.add_argument('--pdbs_json', help='Output pdbs.json', type=str)
    parser.add_argument('--pdbs_csv', help='Output pdbs.csv', type=str)
    parser.add_argument('--domains_html', help='Output domains.html', type=str)
    parser.add_argument('--domains_json', help='Output domains.json', type=str)
    parser.add_argument('--domains_csv', help='Output domains.csv', type=str)
    parser.add_argument('--sample_html', help='Output sample.html', type=str)
    parser.add_argument('--sample_json', help='Output sample.json', type=str)
    parser.add_argument('--sample_csv', help='Output sample.csv', type=str)
    args = parser.parse_args()
    return vars(args)

def format_pdbs_html(family: dict, file: str):
    with open(file, 'w') as w:
        print('<table>', file=w)
        print('  <tr><th>PDB</th></tr>', file=w)
        for pdb, doms in family.items():
            print(f'  <tr><td>{pdb}</td></tr>', file=w)
        print('</table>', file=w)

def format_pdbs_csv(family: dict, file: str):
    pdb_list = list(family.keys())
    with open(file, 'w') as w:
        print('pdb', file=w)
        for pdb in pdb_list:
            print(pdb, file=w)

def format_pdbs_json(family: dict, file: str):
    pdb_list = list(family.keys())
    with open(file, 'w') as w:
        json.dump(pdb_list, w, indent=2)


DOMAIN_FIELDS = [
        ('Domain', 'domain'), 
        ('PDB', 'pdb'), 
        ('Chain', 'chain_id'), 
        ('Ranges', 'ranges'), 
        ('Chain (auth)', 'auth_chain_id'), 
        ('Ranges (auth)', 'auth_ranges'), 
    ]

def format_domains_html(domains: List[dict], file: str):
    with open(file, 'w') as w:
        print('<table>', file=w)
        print('  <tr>', file=w)
        for header, field in DOMAIN_FIELDS:
            print(f'    <th>{header}</th>', file=w)
        print('  </tr>', file=w)
        for dom in domains:
            print('  <tr>', file=w)
            for header, field in DOMAIN_FIELDS:
                value = dom[field]
                print(f'    <td>{value}</td>', file=w)
            print('  </tr>', file=w)
        print('</table>', file=w)

def format_domains_csv(domains: List[dict], file: str):
    with open(file, 'w') as w:
        header_line = ';'.join(field for header, field in DOMAIN_FIELDS)
        print(header_line, file=w)
        for dom in domains:
            line = ';'.join(str(dom[field]) for header, field in DOMAIN_FIELDS)
            print(line, file=w)

def format_domains_json(domains: List[dict], file: str):
    with open(file, 'w') as w:
        json.dump(domains, w, indent=2)


def main(input_family_json: str, input_sample_json: str, 
        pdbs_html: Optional[str], pdbs_json: Optional[str], pdbs_csv: Optional[str], 
        domains_html: Optional[str], domains_json: Optional[str], domains_csv: Optional[str], 
        sample_html: Optional[str], sample_json: Optional[str], sample_csv: Optional[str],
        ) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    try:
        with open(Path(input_family_json)) as r:
            family = json.load(r)
    except FileNotFoundError:
        print(f'Warning: {input_family_json} not found', file=sys.stderr)
        family = {}
    domains = [dom for doms in family.values() for dom in doms]
    try:
        with open(Path(input_sample_json)) as r:
            sample = json.load(r)
    except FileNotFoundError:
        print(f'Warning: {input_sample_json} not found', file=sys.stderr)
        sample = []

    if pdbs_html is not None:
        format_pdbs_html(family, pdbs_html)
    if pdbs_json is not None:
        format_pdbs_json(family, pdbs_json)
    if pdbs_csv is not None:
        format_pdbs_csv(family, pdbs_csv)

    if domains_html is not None:
        format_domains_html(domains, domains_html)
    if domains_json is not None:
        format_domains_json(domains, domains_json)
    if domains_csv is not None:
        format_domains_csv(domains, domains_csv)
        
    if sample_html is not None:
        format_domains_html(sample, sample_html)
    if sample_json is not None:
        format_domains_json(sample, sample_json)
    if sample_csv is not None:
        format_domains_csv(sample, sample_csv)

    return None

def _main():
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)

if __name__ == '__main__':
    _main()

# #############################################################################################

