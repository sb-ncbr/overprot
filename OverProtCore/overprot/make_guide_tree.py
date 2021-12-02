import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from .libs import lib_domains
from .libs import lib_alignment


def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Directory with sample.json and structure files', type=Path)
    parser.add_argument('--show_tree', help='Show the guide tree with ete3', action='store_true')
    parser.add_argument('--progress_bar', help='Show progress bar', action='store_true')
    args = parser.parse_args()
    return vars(args)

def main(directory: Path, show_tree: bool = False, progress_bar: bool = False) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    domains = lib_domains.load_domain_list(directory/'sample.json')
    structure_files = [directory/f'{domain.name}.cif' for domain in domains]
    lib_alignment.make_structure_tree_with_merging(structure_files, show_tree=show_tree, progress_bar=progress_bar)
    return None
    

if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)

