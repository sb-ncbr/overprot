'''
Read `directory/sample.json` and create the guide tree from the listed domains.
The structures must be present in `directory` in files named {domain.name}.cif.

Example usage:
    python3  -m overprot.make_guide_tree  --help
'''
from pathlib import Path
from typing import Optional

from .libs import lib_domains
from .libs import lib_alignment
from .libs.lib_cli import cli_command, run_cli_command


@cli_command()
def main(directory: Path, show_tree: bool = False, progress_bar: bool = False) -> Optional[int]:
    '''Read `directory/sample.json` and create the guide tree from the listed domains.
    The structures must be present in `directory` in files named {domain.name}.cif.
    @param  `directory`     Directory with sample.json and structure files.
    @param  `show_tree`     Show the guide tree with ete3.
    @param  `progress_bar`  Show progress bar.
    '''
    domains = lib_domains.load_domain_list(directory/'sample.json')
    structure_files = [directory/f'{domain.name}.cif' for domain in domains]
    lib_alignment.make_structure_tree_with_merging(structure_files, show_tree=show_tree, progress_bar=progress_bar)
    return None
    

if __name__ == '__main__':
    run_cli_command(main)

