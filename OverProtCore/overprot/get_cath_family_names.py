'''
xtract family names from a CATH family names file.
Output as a JSON format prepared for the hierarchical combobox on OverProt web (cath_b_names_options.json).

Example usage:
    python3  -m overprot.get_cath_family_names  -help
'''

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import shutil
from urllib import request
import gzip
from typing import Dict, Any, Optional, Generic, TypeVar, Iterator

from .libs import lib
from .libs.lib_cli import cli_command, run_cli_command

#  CONSTANTS  ################################################################################

CATH_FAMILY_NAMES_URL = 'ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/daily-release/newest/cath-b-newest-names.gz'

K = TypeVar('K')
V = TypeVar('V')

#  FUNCTIONS  ################################################################################

def download_url(url: str, output_file: Path) -> None:
    print(f'Downloading {url}', file=sys.stderr) 
    with request.urlopen(url) as r:
        with open(output_file, 'wb') as w:
            shutil.copyfileobj(r, w)
    print('Download finished.', file=sys.stderr) 

class LayeredDictNode(Generic[K, V]):
    value: Optional[V]
    children: Dict[K, LayeredDictNode]
    def __init__(self, value: Optional[V] = None) -> None:
        self.value = value
        self.children = {}
    def add(self, keys: tuple[K, ...], value: V) -> None:
        if len(keys) == 0:
            self.value = value
        else:
            head = keys[0]
            tail = keys[1:]
            if head not in self.children:
                self.children[head] = LayeredDictNode()
            self.children[head].add(tail, value)
    def __str__(self, indent: int = 0) -> str:
        return '\n'.join(self._str_lines())
    def _str_lines(self, key: str = '', indent: int = 0) -> Iterator[str]:
        yield f'{"    "*indent}{key}: {self.value}'
        for key, child in self.children.items():
            yield from child._str_lines(key=key, indent=indent+1)
    def to_list(self, key: Optional[K] = None, separator: str = '.') -> list:
        if key is not None:
            result = []
            result.append(str(key))
            if (self.value is not None and self.value != '') or len(self.children) > 0:
                result.append(str(self.value))
            if len(self.children) > 0:
                result.append([child.to_list(key=key+separator+str(child_key), separator=separator) for child_key, child in self.children.items()])
        else:
            result = [child.to_list(key=str(child_key), separator=separator) for child_key, child in self.children.items()]
        return result

def layered_dict_to_options(dictio: dict[str, LayeredDictNode[str, str]], key_prefix: str = '', depth: int = 1) -> list:
    result = []
    for key, node in dictio.items():
        layered_key = key_prefix + key
        if depth < 4:
            layered_key += '.'
        name = node.value
        children = layered_dict_to_options(node.children, key_prefix=layered_key, depth=depth+1)
        nodel = [layered_key]
        if name or children:
            nodel.append(name)
        if children:
            nodel.append(children)
        result.append(nodel)
    return result


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cath_family_names', help='CATH domain list file (like ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/daily-release/newest/cath-b-newest-names.gz)', type=Path)    
    parser.add_argument('--download', help='Download the CATH domain list file and save it in cath_domain_list', action='store_true')
    parser.add_argument('--url', help=f'Specify URL for downloading CATH family names file (only useful with --download), default: {CATH_FAMILY_NAMES_URL}', type=str, default=CATH_FAMILY_NAMES_URL)
    parser.add_argument('-o', '--output', help='Specify output file instead of stdout', type=Path, default=None)
    args = parser.parse_args()
    return vars(args)

@cli_command()
def main(cath_family_names: Path, download: bool = False, url: str = CATH_FAMILY_NAMES_URL, 
         output: Optional[Path] = None) -> Optional[int]:
    '''Extract family names from a CATH family names file.
    Output as a JSON format prepared for the hierarchical combobox on OverProt web (cath_b_names_options.json).
    @param  `cath_family_names`  CATH domain list file.
    @param  `download`           Download the CATH domain list file from `url` and save it in `cath_domain_list`/
    @param  `url`                Specify URL for downloading CATH family names file (only useful with --download).
    @param  `output`             Output file.
    '''
    if download:
        download_url(url, cath_family_names)
    ENCODING = 'utf8'
    if cath_family_names.suffix == '.gz':
        with gzip.open(cath_family_names) as r:
            content = r.read().decode(encoding=ENCODING)
    else:
        with open(cath_family_names, encoding=ENCODING) as r:
            content = r.read()
    family_id_tuples_names = []
    options_dict = LayeredDictNode()
    for line in content.split('\n'):
        line = line.strip()
        if line == '':
            continue
        parts = line.split(maxsplit=1)
        family_id = parts[0]
        family_name = parts[1] if len(parts) > 1 else ''
        family_id_tuple = family_id.split('.')
        family_id_tuples_names.append((family_id_tuple, family_name))
        options_dict.add(family_id_tuple, family_name)
    lib.dump_json(layered_dict_to_options(options_dict.children), output or sys.stdout, minify=True)


if __name__ == '__main__':
    run_cli_command(main)