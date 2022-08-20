from __future__ import annotations
import json
import uuid
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple, Union, Optional, Literal, TypeAlias

from .constants import REFRESH_TIMES, DATA_DIR, JOBS_DIR_COMPLETED, JOBS_DIR_ARCHIVED, JOBS_DIR_FAILED, JOBS_DIR_PENDING, LAST_UPDATE_FILE, DOMAIN_INFO_FILE_TEMPLATE
from .data_caching import DataCacheWithWatchfiles
from .searching import Searcher, FamilyId, DomainId, PdbId


DOMAIN_LIST_FILE = DATA_DIR/'db'/'domain_list.csv'
PDB_LIST_FILE = DATA_DIR/'db'/'pdbs.txt'
EXAMPLE_DOMAINS_FILE = DATA_DIR/'db'/'cath_example_domains.csv'
FAMILY_LIST_FILE = DATA_DIR/'db'/'families.txt'


class ResponseTuple(NamedTuple):
    '''Nice syntax for a HTTP response tuple expected by flask'''
    response: Union[str, dict]
    status: Optional[int] = None
    headers: Union[list, dict, None] = None

    @classmethod
    def plain(cls, response, status=None, headers=None) -> 'ResponseTuple':
        '''Create a plain-text response'''
        if headers is None:
            headers = {'Content-Type': 'text/plain; charset=utf-8'}
        else:
            headers['Content-Type'] = 'text/plain; charset=utf-8'
        return cls(str(response), status, headers)


class DomainInfo(NamedTuple):
    '''Represents basic information about a protein domain'''
    domain: str
    pdb: str
    chain_id: str
    ranges: str
    auth_chain_id: str
    auth_ranges: str
    family: str

    def format_chain(self) -> str:
        if self.chain_id == self.auth_chain_id:
            return self.chain_id
        else:
            return f'{self.chain_id} [auth {self.auth_chain_id}]'
    def format_ranges(self) -> str:
        if self.ranges == self.auth_ranges:
            return self.ranges
        else:
            return f'{self.ranges} [auth {self.auth_ranges}]'
    def format_auth_chain(self) -> str:
        if self.chain_id == self.auth_chain_id:
            return ''
        else:
            return f'[auth {self.auth_chain_id}]'
    def format_auth_ranges(self) -> str:
        if self.chain_id == self.auth_chain_id:
            return ''
        else:
            return f'[auth {self.auth_chain_id}]'


def get_domain_info(domain_id: DomainId) -> DomainInfo:
    path = DOMAIN_INFO_FILE_TEMPLATE.format(domain=domain_id, domain_middle=domain_id[1:3])
    try: 
        js = json.loads(Path(path).read_text())
        return DomainInfo(**js)
    except OSError:
        return DomainInfo(domain_id, '?', '?', '?', '?', '?', '?')

def get_uuid() -> uuid.UUID:
    return uuid.uuid4()

def calculate_time_to_refresh(elapsed_time: timedelta) -> int:
    elapsed_seconds = int(elapsed_time.total_seconds())
    try:
        next_refresh = next(t for t in REFRESH_TIMES if t > elapsed_seconds)
        return next_refresh - elapsed_seconds
    except StopIteration:
        return -elapsed_seconds % REFRESH_TIMES[-1]

def family_exists(family_id: FamilyId) -> bool:
    '''More efficient than using Searcher (reads less data)'''
    return family_id in FAMILY_SET_CACHE.value

FamilyInfoKey = Literal['family_id', 'n_pdbs', 'n_domains', 'n_sample', 'n_sample_without_obsoleted']
def get_family_info(family_id: str) -> dict[FamilyInfoKey, str]:
    SEP = ':'
    result: dict[FamilyInfoKey, str] = {}
    try:
        with open(DATA_DIR/'db'/'family'/'lists'/family_id/'family_info.txt') as f:
            for line in f:
                if SEP in line:
                    key, value = line.split(SEP, maxsplit=1)
                    result[key.strip()] = value.strip()  # type: ignore
        return result
    except OSError:
        return {}

def get_family_info_for_job(job_id: str) -> dict[FamilyInfoKey, str]:
    SEP = ':'
    result: dict[FamilyInfoKey, str] = {}
    try:
        family_info_file = get_job_file(job_id, 'lists', 'family_info.txt')
    except FileNotFoundError:
        return {}
    with open(family_info_file) as f:
        for line in f:
            if SEP in line:
                key, value = line.split(SEP, maxsplit=1)
                result[key.strip()] = value.strip()  # type: ignore
    return result

def get_job_file(job_id: str, *path_parts: str) -> Path:
    '''Get path to file jobs/*/job_id/*path_parts of given job, where * can be Completed, Archived, Failed, or Pending etc. '''
    for db_dir in (JOBS_DIR_COMPLETED, JOBS_DIR_ARCHIVED, JOBS_DIR_FAILED, JOBS_DIR_PENDING):
        path = Path(db_dir, job_id, *path_parts)
        if path.exists():
            return path
    raise FileNotFoundError('/'.join(['...', job_id, *path_parts]))


def _get_family_set() -> set[FamilyId]:
    with open(FAMILY_LIST_FILE) as f:
        result = set()
        for line in f:
            family_id = line.strip()
            if family_id != '':
                result.add(family_id)
    return result

def _get_example_domains() -> dict[FamilyId, DomainId]:
    SEPARATOR = ';'
    with open(EXAMPLE_DOMAINS_FILE) as f:
        result = {}
        for line in f:
            line = line.strip()
            family, example = line.split(SEPARATOR)
            result[family] = example
    return result

def _get_last_update() -> str:
    try:
        return Path(LAST_UPDATE_FILE).read_text().strip()
    except OSError:
        return '???'

FAMILY_SET_CACHE = DataCacheWithWatchfiles(_get_family_set, [FAMILY_LIST_FILE])
EXAMPLE_DOMAINS_CACHE = DataCacheWithWatchfiles(_get_example_domains, [EXAMPLE_DOMAINS_FILE])
LAST_UPDATE_CACHE = DataCacheWithWatchfiles(_get_last_update, [LAST_UPDATE_FILE])
SEARCHER_CACHE = DataCacheWithWatchfiles(lambda: Searcher(DOMAIN_LIST_FILE, pdb_list_txt=PDB_LIST_FILE), [DOMAIN_LIST_FILE, PDB_LIST_FILE])
