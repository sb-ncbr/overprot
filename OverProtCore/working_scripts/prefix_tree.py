from __future__ import annotations
import sys
from pprint import pprint
from pathlib import Path
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypeVar, Generic, NamedTuple, TypeAlias, Any, Iterator, Iterable

T = TypeVar('T')

@dataclass
class Match(Generic[T]):
    key: str
    value: T

KeyStr: TypeAlias = str

# PrefixTreeNode: TypeAlias = tuple[Match[T]|None, dict[KeyStr, tuple]]  # The inner tuple is again PrefixTreeNone but I don't know how solve cyclic dependency

@dataclass
class PrefixTree(Generic[T]):
    match: Match[T]|None = field(default=None)
    root: dict[str, PrefixTree] = field(default_factory = dict)
    def add(self, key: KeyStr, value: T, offset: int = 0) -> None:
        if offset == len(key):
            self.match = Match(key, value)
        else:
            letter = key[offset]
            if letter not in self.root:
                self.root[letter] = PrefixTree()
            self.root[letter].add(key, value, offset=offset+1)
    @classmethod
    def from_items(cls, items: Iterable[tuple[KeyStr, T]]) -> PrefixTree[T]:
        result: PrefixTree[T] = PrefixTree()
        for key, value in sorted(items):
            result.add(key, value)
        return result
    def json(self):
        m = self.match.value if self.match != None else None  # type: ignore
        r = {letter: subtree.json() for letter, subtree in self.root.items()}
        return (m, r)
    def _gen_print_lines(self, depth: int, prefix: list[str], sep: str) -> Iterator[str]:
        if self.match is not None:
            yield f"{' ' * (depth - len(prefix))}{''.join(prefix)}{sep}{self.match.value}"
            prefix = []
        for letter, sub in self.root.items():
            prefix.append(letter)
            yield from sub._gen_print_lines(depth + 1, prefix, sep)
            prefix = []
    def pprint(self, sep = ':', file=sys.stdout) -> None:
        for line in self._gen_print_lines(0, [], sep):
            print(line, file=file)

def get_lines(filename: Path|str, lstrip = False, rstrip = False, rstrip_newline = False, nonempty = False) -> Iterator[str]:
    with open(filename) as f:
        for line in f:
            if lstrip:
                line = line.lstrip()
            if rstrip:
                line = line.rstrip()
            if rstrip_newline:
                line = line.rstrip('\n')
            if nonempty and line == '':
                continue
            yield line


if __name__ == '__main__':
    # trie: PrefixTree[int] = PrefixTree()
    # with open('/home/adam/Workspace/Python/OverProt/docker_mount/data/db/pdbs.txt') as f:
    #     for line in f:
    #         line = line.rstrip()
    #         if line != '':
    #             trie.add(line, len(line))
    # lines = get_lines('/home/adam/Workspace/Python/OverProt/docker_mount/data/db/pdbs.txt', rstrip=True, nonempty=True)
    # trie = PrefixTree.from_items((line, 0) for line in lines)

    # DIR = Path('/home/adam/Workspace/Python/OverProt/data/trie')
    # items = []
    # for line in get_lines(DIR/'cath_b.names.20201021', rstrip_newline=True, nonempty=True):
    #     id, name = line.split(' ', maxsplit=1)
    #     name = ' '.join(name.split()).upper()  # normalize whitespace and case
    #     while len(name)>=3:
    #         items.append((name, id))
    #         name = name[1:]
    # trie = PrefixTree.from_items(items)  # auto sorts
    # # trie = PrefixTree.from_items((line, 0) for line in get_lines('/home/adam/Workspace/Python/OverProt/data/cath_b.names.20201021', rstrip=True, nonempty=True))
    # # trie.add('ahojki', 6)
    # # trie.add('ahojte', 6)
    # # trie.add('cau', 3)
    # # trie.add('cuss', 4)
    # with open(DIR/'cath_b_names.fulltrie.txt', 'w') as w:
    #     trie.pprint(sep='\t', file=w)
    # with open(DIR/'cath_b_names.fulltrie.json', 'w') as w:
    #     json.dump(trie.json(), w, separators=(',',':'))
    # Path('/home/adam/Workspace/Python/OverProt/docker_mount/data/db/domain_list.json').read_text()
    lines = []
    parts = []
    offsets = [0]
    linedict = {}
    with open('/home/adam/Workspace/Python/OverProt/docker_mount/data/db/domain_list.csv', 'r') as f:
        for line in f:
            # lines.append([])
            # lines.append(line)
            parts = line.split(';')
            # lines.append(parts)
            lines.append( tuple(parts) )
            # parts.extend(parts)
            # offsets.append(len(lines))
            # linedict[line] = parts
    # print(lines)
    # print(offsets)