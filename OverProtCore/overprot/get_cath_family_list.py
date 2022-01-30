'''
Extract family IDs from a CATH family list file.

Example usage:
    python3  -m overprot.get_cath_family_list  ./cath-superfamily-list.txt  --download  --sort_by_size \
        --url 'http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-superfamily-list.txt' \
        --out ./families.txt
'''

import sys
from pathlib import Path
import shutil
from urllib import request
from typing import Optional, Counter

from .libs.lib_io import RedirectIO
from .libs.lib_cli import cli_command, run_cli_command

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

@cli_command()
def main(cath_family_list: Path, download: bool = False, url: str = CATH_FAMILY_LIST_URL, 
         out: Optional[Path] = None, sort_by_size: bool = False) -> Optional[int]:
    '''Extract family IDs from a CATH family list file.
    Print the family IDs separated by newline.
    @param  `cath_family_list`  CATH family list file.
    @param  `download`          Download the CATH family list file from `url` and save it in `cath_family_list`.
    @param  `url`               URL for downloading CATH family list file (only useful with `download`).
    @param  `out`               Output file.
    @param  `sort_by_size`      Sort families by number of domains (largest first), rather than alphabetically.
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
    run_cli_command(main)
