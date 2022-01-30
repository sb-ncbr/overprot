'''
Convert domains in `input_family_json` and `input_sample_json` into diferent formats.

Example usage:
    python3  -m overprot.format_domains  --help
'''

import json
import shutil
from pathlib import Path
from typing import Optional, List

from .libs import lib
from .libs.lib_cli import cli_command, run_cli_command

#  FUNCTIONS  ################################################################################

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


#  MAIN  #####################################################################################

@cli_command()
def main(input_family_json: Path, input_sample_json: Path, out_dir: Path,
        per_domain_out_dir: Optional[Path] = None, family_id: Optional[str] = None) -> Optional[int]:
    '''Convert domains in `input_family_json` and `input_sample_json` into diferent formats.
    @param  `input_family_json`   Input family.json.
    @param  `input_sample_json`   Input sample.json.
    @param  `out_dir`             Output directory for per-family files.
    @param  `per_domain_out_dir`  Output directory for per-domain files.
    @param  `family_id`           Family ID to use in the formatted files.
    '''
    with open(input_family_json) as r:
        family: dict[str, list[dict]] = json.load(r)
    domains = [dom for doms in family.values() for dom in doms]
    with open(input_sample_json) as r:
        sample = json.load(r)

    out_dir.mkdir(exist_ok=True)
    shutil.copy(input_family_json, out_dir/'family.json')

    format_pdbs_html(family, out_dir/'pdbs.html', table_id='pdbs', links=True)
    format_pdbs_html(family, out_dir/'pdbs-demo.html', table_id='pdbs', max_rows=MAX_ROWS_IN_DEMO_TABLE, links=True)
    format_pdbs_json(family, out_dir/'pdbs.json')
    format_pdbs_csv(family, out_dir/'pdbs.csv')

    format_domains_html(domains, out_dir/'domains.html', table_id='domains', links=True)
    format_domains_html(domains, out_dir/'domains-demo.html', table_id='domains', max_rows=MAX_ROWS_IN_DEMO_TABLE, links=True)
    format_domains_json(domains, out_dir/'domains.json')
    format_domains_csv(domains, out_dir/'domains.csv')
    
    format_domains_html(sample, out_dir/'sample.html', table_id='sample', links=True)
    format_domains_html(sample, out_dir/'sample-demo.html', table_id='sample', max_rows=MAX_ROWS_IN_DEMO_TABLE, links=True)
    format_domains_json(sample, out_dir/'sample.json')
    format_domains_csv(sample, out_dir/'sample.csv')

    if per_domain_out_dir is not None:
        per_domain_out_dir.mkdir(parents=True, exist_ok=True)
        for domain in domains:
            domain_id = domain['domain']
            domain['family'] = family_id
            format_domain_json(domain, per_domain_out_dir/f'{domain_id}.json')

    return None


if __name__ == '__main__':
    run_cli_command(main)
