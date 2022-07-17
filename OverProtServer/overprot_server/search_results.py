from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from abc import abstractmethod
from typing import Literal, NamedTuple, TypeVar, Iterable
from dataclasses import dataclass, field

from .searching import FamilyId, PdbId, DomainId


class SearchResult(NamedTuple):
    kind: Literal['family', 'pdb', 'domain']
    id: str

T = TypeVar('T')

class NamedList(list[T]):
    _sg_name: str
    _pl_name: str
    def __init__(self, values: Iterable[T] = (), singular_name: str = 'item', plural_name: str = 'items') -> None:
        super().__init__(values)
        self._sg_name = singular_name
        self._pl_name = plural_name
    @property
    def name(self):
        if len(self) == 1:
            return self._sg_name
        else:
            return self._pl_name
    def len(self) -> int:
        return len(self)  # to use in Jinja templates


def _make_families(family_ids: Iterable[FamilyId] = ()) -> NamedList[SearchResult]:
    return NamedList((SearchResult('family', fam) for fam in family_ids), 'family', 'families')
def _make_pdbs(pdb_ids: Iterable[PdbId] = ()) -> NamedList[SearchResult]:
    return NamedList((SearchResult('pdb', pdb) for pdb in pdb_ids), 'PDB entry', 'PDB entries')
def _make_domains(domain_ids: Iterable[DomainId] = ()) -> NamedList[SearchResult]:
    return NamedList((SearchResult('domain', dom) for dom in domain_ids), 'domain', 'domains')

@dataclass
class SearchResults(object):
    families: NamedList[SearchResult] = field(default_factory = lambda: NamedList((), 'family', 'families'))
    pdbs: NamedList[SearchResult] = field(default_factory = _make_pdbs)
    domains: NamedList[SearchResult] = field(default_factory = lambda: NamedList((), 'domain', 'domains'))

    def __len__(self) -> int:
        return len(self.families) + len(self.pdbs) + len(self.domains)
    def len(self) -> int:
        return len(self)  # to use in Jinja templates

    @classmethod
    def from_families(cls, family_ids: Iterable[FamilyId]) -> SearchResults:
        return cls(families = _make_families(family_ids))
    @classmethod
    def from_pdbs(cls, pdb_ids: Iterable[PdbId]) -> SearchResults:
        return cls(pdbs = _make_pdbs(pdb_ids))
    @classmethod
    def from_domains(cls, domain_ids: Iterable[DomainId]) -> SearchResults:
        return cls(domains = _make_domains(domain_ids))
    @classmethod
    def make_fake(cls) -> SearchResults:
        return cls(
            families = NamedList((SearchResult('family', dom) for dom in ['1.10.630.10']), 'family', 'families'),
            pdbs = NamedList((SearchResult('pdb', dom) for dom in ['1bvy', '1tqn']), 'PDB entry', 'PDB entries'),
            domains = NamedList((SearchResult('domain', dom) for dom in ['1bvyA00', '1bvyB00', '1tqnA00', '2nnjA00', '5w97H00', '5w97h00']), 'domain', 'domains'),
        )