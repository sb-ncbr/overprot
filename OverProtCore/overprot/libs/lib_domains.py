import json
from collections import defaultdict, Counter
from typing import Optional, Dict, List, Tuple, Union

from .constants import DOMAIN_NAME, PDB, CHAIN, RANGES, AUTH_CHAIN, AUTH_RANGES, UNIPROT_ID
from .lib import FilePath
from . import lib


COMMENT_SYMBOL = '#'
DOMAIN_FIELD_SEPARATOR = ','
DEFAULT_CHAIN = 'A'


class Domain(dict):
    '''Represents residue ranges of a protein domain'''
    
    @property
    def name(self) -> str: 
        '''Domain name'''
        return self[DOMAIN_NAME]
    
    @property
    def pdb(self) -> str: 
        '''PDB ID'''
        return self[PDB]
        
    @property
    def chain(self) -> str: 
        '''Chain identifier in label_* numbering'''
        return self[CHAIN]
        
    @property
    def ranges(self) -> str: 
        '''Residue ranges in label_* numbering, e.g. "1:50,120:150" '''
        return self[RANGES]
        
    @property
    def auth_chain(self) -> Optional[str]: 
        '''Chain identifier in auth_* numbering'''
        return self[AUTH_CHAIN]
        
    @property
    def auth_ranges(self) -> Optional[str]: 
        '''Residue ranges in auth_* numbering, e.g. "1:50,120:150" '''
        return self[AUTH_RANGES]
        
    @property
    def uniprot(self) -> Optional[str]: 
        '''UniProt accession identifier, e.g. "P14779" '''
        return self.get(UNIPROT_ID, None)  # type: ignore
    
    def __init__(self, *, name: str, pdb: str, chain: str, ranges: str, auth_chain: Optional[str], auth_ranges: Optional[str], uniprot: Optional[str] = None):
        super().__init__()
        self[DOMAIN_NAME] = name
        self[PDB] = pdb
        self[CHAIN] = chain
        self[RANGES] = ranges
        self[AUTH_CHAIN] = auth_chain
        self[AUTH_RANGES] = auth_ranges
        if uniprot is not None:
            self[UNIPROT_ID] = uniprot
    
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'Domain':
        new_domain = cls(name=d[DOMAIN_NAME], pdb=d[PDB], chain=d[CHAIN], ranges=d[RANGES], auth_chain=d[AUTH_CHAIN], auth_ranges=d[AUTH_RANGES], uniprot=d.get(UNIPROT_ID))
        return new_domain


def load_domain_list(filename: Union[FilePath, str]) -> List[Domain]:
    try: 
        result = _load_domain_list_JSON(filename, by_pdb=False)
    except json.JSONDecodeError:
        result = _load_domain_list_TXT(filename, by_pdb=False)
    assert isinstance(result, list)
    return result

def load_domain_list_by_pdb(filename: Union[FilePath, str]) -> Dict[str, List[Domain]]:
    try: 
        result = _load_domain_list_JSON(filename, by_pdb=True)
    except json.JSONDecodeError:
        result = _load_domain_list_TXT(filename, by_pdb=True)
    assert isinstance(result, dict)
    return result

def save_domain_list(domains: Union[List[Domain], Dict[str, List[Domain]]], filename: Union[FilePath, str], by_pdb: Optional[bool] = None) -> None:
    assert isinstance(domains, (list, dict))
    if isinstance(domains, list) and by_pdb == True:
        domains = _group_domains_by_pdb(domains)
    elif isinstance(domains, dict) and by_pdb == False:
        domains = _ungroup_domains_by_pdb(domains)
    lib.dump_json(domains, filename)

# def load_domain_list_by_pdb(filename: Union[FilePath, str]) -> Dict[str, List[Domain]]:
#     with open(filename) as f:
#         obj = json.load(f)
#     assert isinstance(obj, dict)
#     domains_by_pdb = { pdb: [Domain.from_dict(dom) for dom in doms] for pdb, doms in sorted(obj.items()) }
#     return domains_by_pdb

def _load_domain_list_JSON(filename: Union[FilePath, str], by_pdb: bool) -> Union[List[Domain], Dict[str, List[Domain]]]:
    with open(filename) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        domains = [Domain.from_dict(dom) for dom in obj]
        if by_pdb:
            return _group_domains_by_pdb(domains)
        else:
            return domains
    elif isinstance(obj, dict):
        domains_by_pdb = { pdb: [Domain.from_dict(dom) for dom in doms] for pdb, doms in sorted(obj.items()) }
        if by_pdb:
            return domains_by_pdb
        else:
            return _ungroup_domains_by_pdb(domains_by_pdb)
    else:
        raise TypeError('JSON file must contain List[Domain] or Dict[str, List[Domain]]')

def _load_domain_list_TXT(filename: Union[FilePath, str], by_pdb: bool) -> Union[List[Domain], Dict[str, List[Domain]]]:
    domains = []
    domain_counter: Dict[Tuple[str, str], int] = Counter()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line != '' and not line.startswith(COMMENT_SYMBOL):
                pdb, chain, ranges = line.split(DOMAIN_FIELD_SEPARATOR, maxsplit=2)
                pdb = pdb.strip()
                chain = chain.strip()
                ranges = ranges.strip()
                domain_counter[(pdb, chain)] += 1
                c = domain_counter[(pdb, chain)]
                name = f'{pdb}_{chain}_{c:02d}'
                # name = DOMAIN_FIELD_SEPARATOR.join((pdb, chain, ranges)).replace(':', '_')  # : is not allowed in filenames in some file systems
                domain = Domain(name=name, pdb=pdb, chain=chain, ranges=ranges, auth_chain=None, auth_ranges=None)
                domains.append(domain)
    if by_pdb:
        return _group_domains_by_pdb(domains)
    else:
        return domains

def _group_domains_by_pdb(domains: List[Domain]) -> Dict[str, List[Domain]]:
    result = defaultdict(list)
    for dom in domains:
        result[dom.pdb].append(dom)
    return dict(result)

def _ungroup_domains_by_pdb(domains_by_pdb: Dict[str, List[Domain]]) -> List[Domain]:
    return [dom for doms in domains_by_pdb.values() for dom in doms]

