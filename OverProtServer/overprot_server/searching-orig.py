from __future__ import annotations
from pathlib import Path
from typing import TypeAlias


PdbId: TypeAlias = str
CaseInsensitivePdbId: TypeAlias = str
DomainId: TypeAlias = str
CaseInsensitiveDomainId: TypeAlias = str
FamilyId: TypeAlias = str
ChainId: TypeAlias = str
Ranges: TypeAlias = str
UppercaseStr: TypeAlias = str

def _case_neutralize(string: PdbId|CaseInsensitivePdbId|DomainId|CaseInsensitiveDomainId) -> UppercaseStr:
    return string.upper()

class Searcher(object):
    _domains: set[DomainId]
    _pdbs: set[PdbId]
    _families: set[FamilyId]
    _standard_pdbs: dict[UppercaseStr, PdbId]
    _standard_domains: dict[UppercaseStr, DomainId]
    _pdb_to_domains_families: dict[PdbId, list[tuple[DomainId, FamilyId]]]
    _domain_to_family: dict[DomainId, FamilyId]
    _domain_to_pdb: dict[DomainId, PdbId]
    _domain_to_chain_ranges: dict[DomainId, tuple[ChainId, Ranges]]
    
    def __init__(self, domain_list_csv: Path, pdb_list_txt: Path|None = None) -> None:
        SEPARATOR = ';'
        self._domains = set()
        self._pdbs = set()
        self._families = set()
        self._standard_pdbs = {}  # map uppercase to standard
        self._standard_domains = {}  # map uppercase to standard
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
                        self._standard_pdbs[_case_neutralize(pdb)] = pdb
        with open(domain_list_csv) as f:
            for line in f:
                line = line.strip()
                if line != '':
                    family, domain, pdb, chain, ranges = line.split(SEPARATOR)
                    self._domains.add(domain)
                    self._pdbs.add(pdb)
                    self._families.add(family)
                    self._standard_pdbs[_case_neutralize(pdb)] = pdb
                    self._standard_domains[_case_neutralize(domain)] = domain
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
        return self._standard_pdbs.get(pdb.upper())
    def search_domain(self, domain: CaseInsensitiveDomainId) -> DomainId|None:
        return self._standard_domains.get(domain.upper())

    def get_domains_families_for_pdb(self, pdb: PdbId) -> list[tuple[DomainId, FamilyId]]:
        return self._pdb_to_domains_families.get(pdb, [])
    def get_family_for_domain(self, domain: DomainId) -> FamilyId:
        return self._domain_to_family.get(domain, '?')
    def get_pdb_for_domain(self, domain: DomainId) -> PdbId:
        return self._domain_to_pdb.get(domain, '?')
    def get_chain_ranges_for_domain(self, domain: DomainId) -> tuple[ChainId, Ranges]:
        return self._domain_to_chain_ranges.get(domain, ('?', '?'))
    