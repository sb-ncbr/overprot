from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
from typing import Generic, TypeVar, Callable, Optional


class Searcher(object):
    domains: set[str]
    pdbs: set[str]
    families: set[str]
    pdb2domains: dict[str, list[tuple[str, str]]]  # pdb -> [(domain, family)*]
    domain2family: dict[str, str]  # domain -> family
    domain2pdb: dict[str, str]  # domain -> family

    def __init__(self, domain_list_txt: Path) -> None:
        SEPARATOR = ';'
        self.domains = set()
        self.pdbs = set()
        self.families = set()
        self.regular_pdbs = {}  # map uppercase to standard
        self.regular_domains = {}  # map uppercase to standard
        self.pdb2domains = {}
        self.domain2family = {}
        self.domain2pdb = {}
        self.domain2chain_ranges = {}
        with open(domain_list_txt) as f:
            for line in f:
                line = line.strip()
                if line != '':
                    family, domain, pdb, chain, ranges = line.split(SEPARATOR)
                    self.domains.add(domain)
                    self.pdbs.add(pdb)
                    self.families.add(family)
                    self.regular_pdbs[pdb.upper()] = pdb
                    self.regular_domains[domain.upper()] = domain
                    if pdb not in self.pdb2domains:
                        self.pdb2domains[pdb] = []
                    self.pdb2domains[pdb].append((domain, family))
                    self.domain2family[domain] = family
                    self.domain2pdb[domain] = pdb
                    self.domain2chain_ranges[domain] = (chain,ranges)
    def has_pdb(self, pdb: str) -> bool:
        return pdb in self.pdbs
    def has_domain(self, domain: str) -> bool:
        return domain in self.domains
    def has_family(self, family: str) -> bool:
        return family in self.families

    def search_pdb(self, pdb: str) -> str|None:
        return self.regular_pdbs.get(pdb.upper())
    def search_domain(self, domain: str) -> str|None:
        return self.regular_domains.get(domain.upper())

    def get_domains_families_for_pdb(self, pdb: str) -> list[tuple[str, str]]:
        return self.pdb2domains.get(pdb, [])
    def get_family_for_domain(self, domain: str) -> str:
        return self.domain2family.get(domain, '?')
    def get_pdb_for_domain(self, domain: str) -> str:
        return self.domain2pdb.get(domain, '?')
    def get_chain_ranges_for_domain(self, domain: str) -> tuple[str, str]:
        return self.domain2chain_ranges.get(domain, ('?', '?'))

