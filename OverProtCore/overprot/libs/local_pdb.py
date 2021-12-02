import datetime
import re
from re import match
from collections import defaultdict, namedtuple, Counter  # change namedtuple to typing.NamedTuple
from typing import Optional, Sequence, Tuple, List, Dict, Any, Literal

from . import lib

# List of entries: ftp://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/ls-lR, between ./all/mmCIF:\ntotal 0 and empty line

class LocalPDB:
    def __init__(self, directory: str, create_if_not_exists: bool = False):
        raise NotImplementedError

    def __enter__(self):
        '''Load information from the directory to this object.'''
        raise NotImplementedError
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        '''Save information from this object to the directory.'''
        raise NotImplementedError

    @classmethod
    def initialize_local_repo(cls):
        '''Create local directory with PDB mirror'''
        raise NotImplementedError

    def update_status(self):
        '''Update the list of PDB entries and their last update time etc.'''
        raise NotImplementedError

    def update_all_data(self):
        '''Update all structure files according to the current status. Do not update the status itself.'''
        raise NotImplementedError

    def get_entry(self, pdbid: str) -> str:
        '''Update the structure if it has changed and return the filename of the structure file.'''
        raise NotImplementedError


def parse_ls_lR(filename: str) -> Dict[str, List[Tuple[str, str]]]:
    '''Return dictionary {directory: [(file, timestamp), ...], ...}'''
    RE_DIR_HEADER = re.compile('^(\..*):')
    RE_TOTAL = re.compile('^total [0-9]+')
    current_dir = ''
    lines_by_dir: Dict[str, List[Tuple[str, str]]] = {}
    with open(filename) as r:
        for line in r:
            line = line.rstrip()
            header_match = RE_DIR_HEADER.match(line)
            if line == '':
                pass
            elif RE_TOTAL.match(line) is not None:
                pass
            elif header_match is not None:
                # Directory header line
                current_dir = header_match.group(1)
                assert current_dir not in lines_by_dir, f'Duplicate dir: {current_dir}'
                lines_by_dir[current_dir] = []
            else:
                # File record line
                assert current_dir != '', f'Unexpected format, files listed before the first directory: {filename}'
                permissions, n_links, owner, group, size, t1, t2, t3, file, *_ = line.split()
                timestamp = ls_time_to_iso(' '.join((t1, t2, t3)))
                lines_by_dir[current_dir].append((file, timestamp))
    return lines_by_dir

def make_month_dict() -> Dict[str, int]:
    month_names = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'
    month_dict = {}
    for i, name in enumerate(month_names.split(), 1):
        month_dict[name[:3]] = i
        month_dict[str(i)] = i
        month_dict[f'{i:02}'] = i
    return month_dict

MONTH_DICT = make_month_dict()




def ls_time_to_iso(ls_time_string: str) -> str:
    '''Convert time from 'May 1 2020' or 'May 1 10:30' (as from ls -l) to '2020-05-01'.'''
    s_month, s_day, s_year_or_time = ls_time_string.split()
    month = MONTH_DICT[s_month[:3].title()]
    day = int(s_day)
    try:
        year = int(s_year_or_time)
    except ValueError:
        year = datetime.datetime.now().year
    return f'{year:04}-{month:02}-{day:02}'

def test():
    dic = parse_ls_lR('/home/adam/Downloads/ls-lR-20201019')
    files = dic['./all/mmCIF']
    for file, timestamp in sorted(files):
        print(file.split('.')[0], timestamp)

if __name__ == '__main__':
    test()