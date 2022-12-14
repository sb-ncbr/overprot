'''
Download and extract the list of PDB entries.
All lines before a line starting with --- are treated as header.
PDB ID is selected as the first column (whitespace-separated) in the file.

Example usage:
    python3  -m overprot.get_pdb_entry_list  --url 'http://ftp.ebi.ac.uk/pub/databases/pdb/derived_data/index/resolu.idx'  --out ./pdbs.txt
'''

from __future__ import annotations
from pathlib import Path
from urllib import request
from typing import Optional

from .libs import lib
from .libs.lib_io import RedirectIO
from .libs.lib_cli import cli_command, run_cli_command


PDB_ENTRY_LIST_URL = 'http://ftp.ebi.ac.uk/pub/databases/pdb/derived_data/index/resolu.idx'
HEADER_DELIMITER = '---'


@cli_command()
def main(url: str = PDB_ENTRY_LIST_URL, out: Optional[Path] = None) -> Optional[int]:
    '''Download and extract the list of PDB entries.
    All lines before a line starting with --- are treated as header.
    PDB ID is selected as the first column (whitespace-separated) in the file.
    @param  `url`  URL for downloading PDB entry list file.
    @param  `out`  Output file.
    '''
    with request.urlopen(url) as r:
        content: bytes = r.read() 
    text = content.decode('utf8')
    lines = text.splitlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    data_start_line_index = lib.first_index_where(lines, lambda line: line.startswith(HEADER_DELIMITER)) + 1
    pdbs = []
    for line in lines[data_start_line_index:]:
        pdb, *_  = line.split(maxsplit=1)
        pdb = pdb.lower()
        pdbs.append(pdb)
    pdbs = sorted(set(pdbs))  # Remove duplicities from PDB_ENTRY_LIST_URL
    with RedirectIO(stdout=out):
        print(*pdbs, sep='\n')
    return 0


if __name__ == '__main__':
    run_cli_command(main)