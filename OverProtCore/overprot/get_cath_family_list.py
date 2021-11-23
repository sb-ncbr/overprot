'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import argparse
import sys
import shutil
from urllib import request
# from collections import Counter
from typing import Dict, Any, Optional, Union, Counter

from .libs import lib
from .libs.lib import FilePath

#  CONSTANTS  ################################################################################

CATH_DOMAIN_LIST_URL = 'ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt'

#  FUNCTIONS  ################################################################################

def download_url(url: str, output_file: FilePath) -> None:
    print(f'Downloading {url}', file=sys.stderr) 
    with request.urlopen(url) as r:
        with output_file.open('wb') as w:
            shutil.copyfileobj(r, w)
    print('Download finished.', file=sys.stderr) 

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cath_domain_list', help='CATH domain list file (like ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt)', type=str)    
    parser.add_argument('--download', help='Download the CATH domain list file and save it in cath_domain_list', action='store_true')
    parser.add_argument('--url', help=f'Specify URL for downloading CATH domain list file (only useful with --download), default: {CATH_DOMAIN_LIST_URL}', type=str, default=CATH_DOMAIN_LIST_URL)
    parser.add_argument('-o', '--output', help='Specify output file instead of stdout', type=str, default=None)
    parser.add_argument('-s', '--sort_by_size', help='Sort families by number of domains (largest first), rather than alphabetically', action='store_true')
    args = parser.parse_args()
    return vars(args)


def main(cath_domain_list: Union[FilePath, str], download: bool = False, url: str = CATH_DOMAIN_LIST_URL, 
         output: Union[FilePath, str, None] = None, sort_by_size: bool = False) -> Optional[int]:
    '''Extract family IDs from a CATH domain list file (like ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt).
    Print the family IDs separated by newline.
    If download==True, first download the file from url to cath_domain_list.
    If sort_by_size==True, sort the families by number of domains (largest first), otherwise sort alphabetically.
    '''
    cath_domain_list = FilePath(cath_domain_list)
    if download:
        download_url(url, cath_domain_list)
    family_sizes: Counter[str] = Counter()
    with cath_domain_list.open() as f:
        for line in f:
            if not line.startswith('#'):
                domain, c, a, t, h, s, o, l, i, d, length, resolution = line.split()
                family = f'{c}.{a}.{t}.{h}'
                family_sizes[family] += 1
    if sort_by_size:
        families = [fam for fam, size in family_sizes.most_common()]
    else:
        families = sorted(family_sizes.keys())
    with lib.RedirectIO(stdout=output):
        print(*families, sep='\n')
    return 0


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)