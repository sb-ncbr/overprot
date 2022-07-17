from __future__ import annotations
from pathlib import Path
from abc import abstractmethod
from typing import TypeAlias, Generic, TypeVar


TExact = TypeVar('TExact')
TInsensitive = TypeVar('TInsensitive', contravariant=True)
TNeutral = TypeVar('TNeutral', covariant=True)

class _BaseReprManager(Generic[TExact, TInsensitive, TNeutral]):
    '''Abstract class for managing object that can be represented in different ways (TInsensitive, e.g. case-insensitive strings 'van de graaff' vs 'van DE GRAAFF') 
    but have a preferred representation (TExact, e.g. exact case-sensitive string 'Van de Graaff),
    all mapping to the same neutral representation (TNeutral, e.g. upper-case string 'VAN DE GRAPH').
    '''
    entries: set[TExact]
    regular_entries: dict[TNeutral, TExact]

    @classmethod
    @abstractmethod
    def neutralize(cls, key: TInsensitive|TExact) -> TNeutral:
        '''Convert representation into a neutral form (e.g. '1TQn' -> '1TQN')'''

    class DuplicateError(ValueError):
        '''Raised when trying to add two equivalent values, e.g. '1tqn' and '1TQN'.'''

    def __init__(self) -> None:
        self.entries = set()
        self.regular_entries = {}

    def __contains__(self, exact_key: TExact) -> bool:
        '''Decide if this exact value (e.g. 1tqn) is here'''
        return exact_key in self.entries
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def add(self, exact_key: TExact) -> None:
        '''Add exact value (e.g. 1tqn)'''
        neutral = self.neutralize(exact_key)
        if neutral in self.regular_entries.keys():
            old_key = self.regular_entries[neutral]
            if exact_key != old_key:
                raise _BaseReprManager.DuplicateError(f'Equivalent objects: cannot add {exact_key!r} because {old_key!r} is already present')
        self.entries.add(exact_key)
        self.regular_entries[neutral] = exact_key
    
    def search(self, insensitive_key: TInsensitive) -> TExact|None:
        '''Find preferred name for this value (e.g. 1TQn -> 1tqn)'''
        return self.regular_entries.get(self.neutralize(insensitive_key))
 
TExactStr = TypeVar('TExactStr', bound=str)
TInsensitiveStr = TypeVar('TInsensitiveStr', bound=str, contravariant=True)
UppercaseStr: TypeAlias = str

class CaseManager(_BaseReprManager[TExactStr, TInsensitiveStr, UppercaseStr]):
    '''Class for managing case-insensitive string (e.g. 'van de graaff' vs 'van DE GRAAFF') 
    which have a preferred representation case-sensitive (e.g. 'Van de Graaff),
    all mapping to the same neutral representation (e.g. 'VAN DE GRAPH').'''
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
    _pdb_to_domains_families: dict[PdbId, list[tuple[DomainId, FamilyId]]]
    _domain_to_family: dict[DomainId, FamilyId]
    _domain_to_pdb: dict[DomainId, PdbId]
    _domain_to_chain_ranges: dict[DomainId, tuple[ChainId, Ranges]]
    
    def __init__(self, domain_list_csv: Path, pdb_list_txt: Path|None = None) -> None:
        SEPARATOR = ';'
        self._domains = CaseManager()
        self._pdbs = CaseManager()
        self._families = set()
        self._pdb_to_domains_families = {}
        self._domain_to_family = {}
        self._domain_to_pdb = {}
        self._domain_to_chain_ranges = {}
        if pdb_list_txt is not None:
            with open(pdb_list_txt) as f:
                for line in f:
                    line = line.strip()
                    if line != '':
                        pdb: PdbId = line
                        self._pdbs.add(pdb)
        with open(domain_list_csv) as f:
            for line in f:
                line = line.strip()
                if line != '':
                    family, domain, pdb, chain, ranges = line.split(SEPARATOR)
                    self._domains.add(domain)
                    self._pdbs.add(pdb)
                    self._families.add(family)
                    if pdb not in self._pdb_to_domains_families:
                        self._pdb_to_domains_families[pdb] = []
                    self._pdb_to_domains_families[pdb].append((domain, family))
                    self._domain_to_family[domain] = family
                    self._domain_to_pdb[domain] = pdb
                    self._domain_to_chain_ranges[domain] = (chain,ranges)
                    
    def has_pdb(self, pdb: PdbId) -> bool:
        return pdb in self._pdbs
    def has_domain(self, domain: DomainId) -> bool:
        return domain in self._domains
    def has_family(self, family: FamilyId) -> bool:
        return family in self._families

    def search_pdb(self, pdb: CaseInsensitivePdbId) -> PdbId|None:
        return self._pdbs.search(pdb)
    def search_domain(self, domain: CaseInsensitiveDomainId) -> DomainId|None:
        return self._domains.search(domain)

    def get_domains_families_for_pdb(self, pdb: PdbId) -> list[tuple[DomainId, FamilyId]]:
        return self._pdb_to_domains_families.get(pdb, [])
    def get_family_for_domain(self, domain: DomainId) -> FamilyId:
        return self._domain_to_family.get(domain, '?')
    def get_pdb_for_domain(self, domain: DomainId) -> PdbId:
        return self._domain_to_pdb.get(domain, '?')
    def get_chain_ranges_for_domain(self, domain: DomainId) -> tuple[ChainId, Ranges]:
        return self._domain_to_chain_ranges.get(domain, ('?', '?'))
    