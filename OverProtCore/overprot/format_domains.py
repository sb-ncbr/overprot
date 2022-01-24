import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

from .libs import lib

#  FUNCTIONS  ################################################################################

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_family_json', help='Input family.json', type=Path)
    parser.add_argument('input_sample_json', help='Input sample.json', type=Path)
    parser.add_argument('--pdbs_html', help='Output pdbs.html', type=Path)
    parser.add_argument('--pdbs_demo_html', help='Output pdbs-demo.html (limited number of rows)', type=Path)
    parser.add_argument('--pdbs_json', help='Output pdbs.json', type=Path)
    parser.add_argument('--pdbs_csv', help='Output pdbs.csv', type=Path)
    parser.add_argument('--domains_html', help='Output domains.html', type=Path)
    parser.add_argument('--domains_demo_html', help='Output domains-demo.html (limited number of rows)', type=Path)
    parser.add_argument('--domains_json', help='Output domains.json', type=Path)
    parser.add_argument('--domains_csv', help='Output domains.csv', type=Path)
    parser.add_argument('--sample_html', help='Output sample.html', type=Path)
    parser.add_argument('--sample_demo_html', help='Output sample-demo.html (limited number of rows)', type=Path)
    parser.add_argument('--sample_json', help='Output sample.json', type=Path)
    parser.add_argument('--sample_csv', help='Output sample.csv', type=Path)
    parser.add_argument('--out_dir', help='Output directory for all output file, overrides all other options', type=Path)
    args = parser.parse_args()
    return vars(args)

def format_pdbs_html(family: dict, file: Path, table_id: Optional[str] = None, max_rows: Optional[int] = None, links: bool = False):
    pdbs = list(family.keys())
    n_total_rows = len(pdbs)
    if max_rows is not None and n_total_rows > max_rows:
        pdbs = pdbs[:max_rows]
        truncated = True
    else:
        truncated = False
    with open(file, 'w') as w:
        if table_id is not None:
            print(f'<table id="{table_id}">', file=w)
        else:
            print('<table>', file=w)
        print(' <thead>', file=w)
        print('  <tr><th colname="pdb">PDB</th></tr>', file=w)
        print(' </thead>', file=w)
        print(' <tbody>', file=w)
        for pdb in pdbs:
            if links:
                print(f'  <tr><td><a href="/pdb/{pdb}">{pdb}</a></td></tr>', file=w)
            else:
                print(f'  <tr><td>{pdb}</td></tr>', file=w)
        print(' </tbody>', file=w)
        print('</table>', file=w)
        if truncated:
            print('<div>', file=w)
            print(f' <button class="btn btn-link btn-load-all">Load all {n_total_rows} PDBs...</button>', file=w)
            print('</div>', file=w)

def format_pdbs_csv(family: dict, file: Path):
    pdb_list = list(family.keys())
    with open(file, 'w') as w:
        print('pdb', file=w)
        for pdb in pdb_list:
            print(pdb, file=w)

def format_pdbs_json(family: dict, file: Path):
    pdb_list = list(family.keys())
    lib.dump_json(pdb_list, file)


DOMAIN_FIELDS = [
        ('Domain', 'domain'), 
        ('PDB', 'pdb'), 
        ('Chain', 'chain_id'), 
        ('Ranges', 'ranges'), 
        ('Chain (auth)', 'auth_chain_id'), 
        ('Ranges (auth)', 'auth_ranges'), 
    ]

MAX_ROWS_IN_DEMO_TABLE = 50

def format_domains_html(domains: List[dict], file: Path, table_id: Optional[str] = None, max_rows: Optional[int] = None, links: bool = False):
    n_total_rows = len(domains)
    if max_rows is not None and n_total_rows > max_rows:
        domains = domains[:max_rows]
        truncated = True
    else:
        truncated = False
    with open(file, 'w') as w:
        if table_id is not None:
            print(f'<table id="{table_id}">', file=w)
        else:
            print('<table>', file=w)
        print(' <thead>', file=w)
        print('  <tr>', file=w)
        for header, field in DOMAIN_FIELDS:
            print(f'   <th colname="{field}">{header}</th>', file=w)
        print('  </tr>', file=w)
        print(' </thead>', file=w)
        print(' <tbody>', file=w)
        for dom in domains:
            print('  <tr>', file=w)
            for header, field in DOMAIN_FIELDS:
                value = dom[field] or ""
                if links and field == 'pdb':
                    print(f'   <td><a href="/pdb/{value}">{value}</a></td>', file=w)
                elif links and field == 'domain':
                    print(f'   <td><a href="/domain/{value}">{value}</a></td>', file=w)
                else:
                    print(f'   <td>{value or ""}</td>', file=w)
            print('  </tr>', file=w)
        print(' </tbody>', file=w)
        print('</table>', file=w)
        if truncated:
            print('<div>', file=w)
            print(f' <button class="btn btn-link btn-load-all">Load all {n_total_rows} domains...</button>', file=w)
            print('</div>', file=w)

def format_domains_csv(domains: List[dict], file: Path):
    with open(file, 'w') as w:
        header_line = ';'.join(field for header, field in DOMAIN_FIELDS)
        print(header_line, file=w)
        for dom in domains:
            line = ';'.join(str(dom[field] or "") for header, field in DOMAIN_FIELDS)
            print(line, file=w)

def format_domains_json(domains: List[dict], file: Path):
    lib.dump_json(domains, file)

def format_domain_json(domain: dict, file: Path):
    lib.dump_json(domain, file)


def main(input_family_json: Path, input_sample_json: Path, 
        pdbs_html: Optional[Path] = None, pdbs_demo_html: Optional[Path] = None, pdbs_json: Optional[Path] = None, pdbs_csv: Optional[Path] = None, 
        domains_html: Optional[Path] = None, domains_demo_html: Optional[Path] = None, domains_json: Optional[Path] = None, domains_csv: Optional[Path] = None, 
        sample_html: Optional[Path] = None, sample_demo_html: Optional[Path] = None, sample_json: Optional[Path] = None, sample_csv: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        per_domain_out_dir: Optional[Path] = None,
        family_id: Optional[str] = None) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    with open(input_family_json) as r:
        family: dict[str, list[dict]] = json.load(r)
    domains = [dom for doms in family.values() for dom in doms]
    with open(input_sample_json) as r:
        sample = json.load(r)

    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
        shutil.copy(input_family_json, out_dir/'family.json')
        pdbs_html = out_dir/'pdbs.html'
        pdbs_demo_html = out_dir/'pdbs-demo.html'
        pdbs_json = out_dir/'pdbs.json'
        pdbs_csv = out_dir/'pdbs.csv'
        domains_html = out_dir/'domains.html'
        domains_demo_html = out_dir/'domains-demo.html'
        domains_json = out_dir/'domains.json'
        domains_csv = out_dir/'domains.csv'
        sample_html = out_dir/'sample.html'
        sample_demo_html = out_dir/'sample-demo.html'
        sample_json = out_dir/'sample.json'
        sample_csv = out_dir/'sample.csv'

    if pdbs_html is not None:
        format_pdbs_html(family, pdbs_html, table_id='pdbs', links=True)
    if pdbs_demo_html is not None:
        format_pdbs_html(family, pdbs_demo_html, table_id='pdbs', max_rows=MAX_ROWS_IN_DEMO_TABLE, links=True)
    if pdbs_json is not None:
        format_pdbs_json(family, pdbs_json)
    if pdbs_csv is not None:
        format_pdbs_csv(family, pdbs_csv)

    if domains_html is not None:
        format_domains_html(domains, domains_html, table_id='domains', links=True)
    if domains_demo_html is not None:
        format_domains_html(domains, domains_demo_html, table_id='domains', max_rows=MAX_ROWS_IN_DEMO_TABLE, links=True)
    if domains_json is not None:
        format_domains_json(domains, domains_json)
    if domains_csv is not None:
        format_domains_csv(domains, domains_csv)
        
    if sample_html is not None:
        format_domains_html(sample, sample_html, table_id='sample', links=True)
    if sample_demo_html is not None:
        format_domains_html(sample, sample_demo_html, table_id='sample', max_rows=MAX_ROWS_IN_DEMO_TABLE, links=True)
    if sample_json is not None:
        format_domains_json(sample, sample_json)
    if sample_csv is not None:
        format_domains_csv(sample, sample_csv)

    if per_domain_out_dir is not None:
        per_domain_out_dir.mkdir(parents=True, exist_ok=True)
        for domain in domains:
            domain_id = domain['domain']
            domain['family'] = family_id
            format_domain_json(domain, per_domain_out_dir/f'{domain_id}.json')

    return None

def _main():
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)

if __name__ == '__main__':
    _main()

# #############################################################################################

