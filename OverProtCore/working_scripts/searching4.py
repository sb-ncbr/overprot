from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from abc import abstractmethod
from typing import TypeAlias, Generic, TypeVar, Iterable, Any


TExact = TypeVar('TExact')
TInsensitive = TypeVar('TInsensitive', contravariant=True)
TNeutral = TypeVar('TNeutral', covariant=True)

K = TypeVar('K')
V = TypeVar('V')

# Index: TypeAlias = int  # index to the offset array
# Offset: TypeAlias = int  # index to the value array

def _make_range_dict(key_vals: list[tuple[K, V]]) -> tuple[ dict[K,int], list[int], list[V] ]:
    SENTINEL: Any = None  # must not be in keys
    NEVER_VALUE: Any = None 
    key_vals.sort()
    key_vals.append((SENTINEL, NEVER_VALUE))  # sentinel
    index: dict[K,int] = {}
    offsets: list[int] = []
    all_values: list[V] = []
    current_key = SENTINEL
    for i, (key, val) in enumerate(key_vals):
        if key != current_key:
            index[key] = len(offsets)
            offsets.append(i)
            current_key = key
        all_values.append(val)
    index.pop(SENTINEL)
    return index, offsets, all_values
                

class _BaseReprManager(Generic[TExact, TInsensitive, TNeutral]):
    '''Abstract class for managing entries with an exact representation (TExact, e.g. case-sensitive strings 'Pole' != 'pole'), 
    but can be represented in a loose way (TInsensitive, e.g. case-insensitive strings 'POLE' == 'Pole' == 'pole') 
    all mapping to the same neutral representation (TNeutral, e.g. upper-case string 'POLE').
    '''
    key_set: set[TExact]
    key_list: list[TExact]
    index: list[int]
    offsets: list[int]

    @classmethod
    @abstractmethod
    def neutralize(cls, key: TInsensitive|TExact) -> TNeutral:
        '''Convert representation into a neutral form (e.g. '1TQn' -> '1TQN')'''

    class DuplicateError(ValueError):
        '''Raised when trying to add two equivalent values, e.g. '1tqn' and '1TQN'.'''

    def __init__(self, keys: set[TExact]) -> None:
        self.key_set = keys
        neutral_exact_list = [(self.neutralize(key), key) for key in keys]
        self.index, self.offsets, self.key_list = _make_range_dict(neutral_exact_list)

    def __contains__(self, exact_key: TExact) -> bool:
        '''Decide if this exact value (e.g. 1tqn) is here'''
        return exact_key in self.key_set
    
    def __len__(self) -> int:
        return len(self.key_set)
    
    def search(self, insensitive_key: TInsensitive) -> list[TExact]:
        '''Find preferred name for this value (e.g. 1TQn -> 1tqn)'''
        neutral = self.neutralize(insensitive_key)
        try:
            i = self.index[neutral]
            fro = self.offsets[i]
            to = self.offsets[i+1]
            return self.key_list[fro:to]
        except KeyError:
            return []
 
TExactStr = TypeVar('TExactStr', bound=str)
TInsensitiveStr = TypeVar('TInsensitiveStr', bound=str, contravariant=True)
UppercaseStr: TypeAlias = str

class CaseManager(_BaseReprManager[TExactStr, TInsensitiveStr, UppercaseStr]):
    '''Class for managing entries case-sensitive strings(TExactStr, 'Pole' != 'pole'), 
    that can be represented as case-insensitive (TInsensitive, 'POLE' == 'Pole' == 'pole') 
    all mapping to the same neutral representation (TNeutral, upper-case string 'POLE').'''
    @classmethod
    def neutralize(cls, key: TInsensitiveStr|TExactStr) -> UppercaseStr:
        '''Convert string into a case-neutral form (here uppercase, e.g. '1TQn' -> '1TQN')'''
        return key.upper()


PdbId: TypeAlias = str
CaseInsensitivePdbId: TypeAlias = str
DomainId: TypeAlias = str
CaseInsensitiveDomainId: TypeAlias = str
FamilyId: TypeAlias = str
ChainId: TypeAlias = str
Ranges: TypeAlias = str

class Searcher(object):
    _domains: CaseManager[DomainId, CaseInsensitiveDomainId]
    _pdbs: CaseManager[PdbId, CaseInsensitivePdbId]
    _families: set[FamilyId]
    _all_domains: list[DomainId]
    _all_families: list[FamilyId]
    _pdb_range_dict: dict[PdbId, tuple[int,int]]  # maps PDBs to ranges in _all_domains and _all_families
    _domain_to_family: dict[DomainId, FamilyId]
    _domain_to_pdb: dict[DomainId, PdbId]
    _domain_to_chain_ranges: dict[DomainId, tuple[ChainId, Ranges]]
    
    def __init__(self, domain_list_csv: Path, pdb_list_txt: Path|None = None) -> None:
        SEPARATOR = ';'
        self._domain_to_family = {}
        self._domain_to_pdb = {}
        self._domain_to_chain_ranges = {}
        pdbs: set[PdbId] = set()
        doms: set[DomainId] = set()
        fams: set[FamilyId] = set()
        if pdb_list_txt is not None:
            with open(pdb_list_txt) as f:
                for line in f:
                    line = line.strip()
                    if line != '':
                        pdb: PdbId = line
                        pdbs.add(pdb)
        with open(domain_list_csv) as f:
            pdb_dom_fam: list[tuple[PdbId, DomainId, FamilyId]] = list()
            for line in f:
                line = line.strip()
                if line != '':
                    family, domain, pdb, chain, ranges = line.split(SEPARATOR)
                    pdbs.add(pdb)
                    doms.add(domain)
                    fams.add(family)
                    pdb_dom_fam.append((pdb, domain, family))
                    self._domain_to_family[domain] = family
                    self._domain_to_pdb[domain] = pdb
                    self._domain_to_chain_ranges[domain] = (chain,ranges)
            self._pdbs = CaseManager(pdbs)
            self._domains = CaseManager(doms)
            self._families = fams
            self._all_domains, self._all_families, self._pdb_range_dict =  Searcher._make_doms_fams_range_dict(pdb_dom_fam)
    
    @staticmethod
    def _make_doms_fams_range_dict(pdb_dom_fam: list[tuple[PdbId, DomainId, FamilyId]]) -> tuple[
            list[DomainId], list[FamilyId], dict[PdbId, tuple[int,int]] ]:
        pdb_dom_fam.sort()
        pdb_dom_fam.append(('', '', ''))  # sentinel
        working_dom: list[DomainId] = []
        working_fam: list[FamilyId] = []
        range_dict: dict[PdbId, tuple[int,int]] = {}
        current_pdb = ''
        run_start = 0
        for i, (pdb, dom, fam) in enumerate(pdb_dom_fam):
            if pdb != current_pdb:
                range_dict[current_pdb] = (run_start, i)
                current_pdb = pdb
                run_start = i
            working_dom.append(dom)
            working_fam.append(fam)
        range_dict.pop('')
        return working_dom, working_fam, range_dict
                    
    def has_pdb(self, pdb: PdbId) -> bool:
        return pdb in self._pdbs
    def has_domain(self, domain: DomainId) -> bool:
        return domain in self._domains
    def has_family(self, family: FamilyId) -> bool:
        return family in self._families

    def search_pdb(self, pdb: CaseInsensitivePdbId) -> list[PdbId]:
        return sorted(self._pdbs.search(pdb))
    def search_domain(self, domain: CaseInsensitiveDomainId) -> list[DomainId]:
        return sorted(self._domains.search(domain))

    def get_domains_families_for_pdb(self, pdb: PdbId) -> list[tuple[DomainId, FamilyId]]:
        try:
            fro, to = self._pdb_range_dict[pdb]
            doms = self._all_domains[fro:to]
            fams = self._all_families[fro:to]
            return list(zip(doms, fams))
        except KeyError:
            return []
    def get_family_for_domain(self, domain: DomainId) -> FamilyId:
        return self._domain_to_family.get(domain, '?')
    def get_pdb_for_domain(self, domain: DomainId) -> PdbId:
        return self._domain_to_pdb.get(domain, '?')
    def get_chain_ranges_for_domain(self, domain: DomainId) -> tuple[ChainId, Ranges]:
        return self._domain_to_chain_ranges.get(domain, ('?', '?'))
    