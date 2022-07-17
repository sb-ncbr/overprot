from __future__ import annotations
import sys
from pprint import pprint
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
    def _gen_print_lines(self, depth: int, prefix: list[str]) -> Iterator[str]:
        if self.match is not None:
            yield ' ' * (depth - len(prefix)) + ''.join(prefix) + ':' + str(self.match.value)
            prefix = []
        for letter, sub in self.root.items():
            prefix.append(letter)
            yield from sub._gen_print_lines(depth + 1, prefix)
            prefix = []
    def pprint(self, file=sys.stdout) -> None:
        for line in self._gen_print_lines(0, []):
            print(line, file=file)
        


if __name__ == '__main__':
    trie: PrefixTree[int] = PrefixTree()
    with open('/home/adam/Workspace/Python/OverProt/docker_mount/data/db/pdbs.txt') as f:
        for line in f:
            line = line.rstrip()
            if line != '':
                trie.add(line, len(line))
    # trie.add('ahojki', 6)
    # trie.add('ahojte', 6)
    # trie.add('cau', 3)
    # trie.add('cuss', 4)
    trie.pprint()
    # print(json.dumps(trie.json(), separators=(':',',')))