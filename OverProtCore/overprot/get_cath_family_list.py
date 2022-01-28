'''
This Python script extracts family IDs from a CATH family list file 
(like http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-superfamily-list.txt).
Prints the family IDs separated by newline.
If download==True, it first downloads the file.
If sort_by_size==True, it sorts the families by number of domains (largest first), otherwise sorts alphabetically.

Example usage:
    python3  -m overprot.get_cath_family_list  ./cath-superfamily-list.txt  --download  --sort_by_size \
        --url 'http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-superfamily-list.txt' \
        --out ./families.txt
'''

import argparse
import sys
from pathlib import Path
import shutil
from urllib import request
from typing import Dict, Any, Optional, Counter

from .libs import lib
from .libs.lib_io import RedirectIO

#  CONSTANTS  ################################################################################

CATH_FAMILY_LIST_URL = 'http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-superfamily-list.txt'

#  FUNCTIONS  ################################################################################

def download_url(url: str, output_file: Path) -> None:
    print(f'Downloading {url}', file=sys.stderr) 
    with request.urlopen(url) as r:
        with open(output_file, 'wb') as w:
            shutil.copyfileobj(r, w)
    print('Download finished.', file=sys.stderr) 

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cath_family_list', help=f'CATH family list file (like {CATH_FAMILY_LIST_URL})', type=Path)    
    parser.add_argument('--download', help='Download the CATH family list file and save it in cath_family_list', action='store_true')
    parser.add_argument('--url', help=f'Specify URL for downloading CATH family list file (only useful with --download), default: {CATH_FAMILY_LIST_URL}', type=str, default=CATH_FAMILY_LIST_URL)
    parser.add_argument('-o', '--out', help='Specify output file instead of stdout', type=Path, default=None)
    parser.add_argument('-s', '--sort_by_size', help='Sort families by number of domains (largest first), rather than alphabetically', action='store_true')
    args = parser.parse_args()
    return vars(args)


def main(cath_family_list: Path, download: bool = False, url: str = CATH_FAMILY_LIST_URL, 
         out: Optional[Path] = None, sort_by_size: bool = False) -> Optional[int]:
    '''Extract family IDs from a CATH family list file (like http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-superfamily-list.txt).
    Print the family IDs separated by newline.
    If download==True, first download the file from url to cath_family_list.
    If sort_by_size==True, sort the families by number of domains (largest first), otherwise sort alphabetically.
    '''
    if download:
        download_url(url, cath_family_list)
    family_sizes: dict[str, int] = Counter()
    with open(cath_family_list,) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#') and line != '':
                family, n_S35, n_domains, *_ = line.split('\t')
                family_sizes[family] = n_domains
    if sort_by_size:
        families = [fam for fam, size in family_sizes.most_common()]
    else:
        families = sorted(family_sizes.keys())
    with RedirectIO(stdout=out):
        print(*families, sep='\n')
    return 0


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)