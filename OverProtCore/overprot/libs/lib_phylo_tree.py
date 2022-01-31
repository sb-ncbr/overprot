'''Representation of a phylogenetic tree'''

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, List


NO_CHILD = -1  # child number meaning that there is no child (in binary trees)
NO_PARENT = -1  # parent number meaning that there is no parent (in binary trees)

Children = Any  # numpy.ndarray[n,2] of int

@dataclass
class PhyloTree(object):
    name: Optional[str] = None
    distance: Optional[float] = None
    weight: float = 1
    children: 'List[PhyloTree]' = field(default_factory=list)

    @classmethod
    def merge(cls, *subtrees: 'PhyloTree', name=None) -> 'PhyloTree':
        result = cls(name=name, weight=0)
        for subtree in subtrees:
            result.children.append(subtree)
            result.weight += subtree.weight
        return result

    @classmethod
    def _from_children(cls, children: Children, root: int, distances, node_names) -> 'PhyloTree':
        n = len(children)
        assert -n <= root < n
        if root < 0:
            root += n
        if children[root, 0] == NO_CHILD:  # it is a leaf
            result = PhyloTree()
        else:
            dist = distances[root]
            subtree1 = cls._from_children(children, children[root, 0], distances, node_names)
            subtree2 = cls._from_children(children, children[root, 1], distances, node_names)
            result = cls.merge(subtree1, subtree2)
            WEIGHT_DISTANCES = True
            if WEIGHT_DISTANCES:
                subtree1.distance = dist * subtree2.weight / result.weight
                subtree2.distance = dist * subtree1.weight / result.weight
            else:
                subtree1.distance = dist / 2
                subtree2.distance = dist / 2
        if node_names is not None and root < len(node_names):
            result.name = node_names[root]
        else:
            result.name = f'#{root}'
        return result
    
    @classmethod
    def from_children(cls, children: Children, roots: Iterable[int] = (-1,), distances=defaultdict(lambda: 1), node_names=None) -> 'PhyloTree':
        n = len(children)
        trees = [cls._from_children(children, root, distances, node_names) for root in roots]
        if len(trees) == 1:
            return trees[0]
        elif len(trees) > 1:
            return cls.merge(*trees)
        else:
            raise ValueError('There must be at least one root index')
    
    def _str_tokens(self, end='') -> Iterator[Any]:
        '''Represent the tree in Newick format.'''
        if len(self.children) > 0:
            yield '('
            for i, subtree in enumerate(self.children):
                if i > 0:
                    yield ','
                yield from subtree._str_tokens()
            yield ')'
        if self.name is not None:  # and self.name[0] != '#':
            yield self.name
        if self.weight != 1:
            yield f'[{self.weight}]'
        if self.distance is not None:
            yield ':'
            yield str(self.distance)
        yield end

    def __str__(self, end=';') -> str:
        '''Represent the tree in Newick format.'''
        return ''.join(self._str_tokens(end=end))
        
    def _add_to_arrays(self, names, weights, dists, parents, children) -> int:
        if len(self.children) == 0:
            i_left_child = NO_CHILD
            i_right_child = NO_CHILD
            index = len(names)
        elif len(self.children) == 2:
            i_left_child = self.children[0]._add_to_arrays(names, weights, dists, parents, children)
            i_right_child = self.children[1]._add_to_arrays(names, weights, dists, parents, children)
            index = len(names)
            parents[i_left_child] = parents[i_right_child] = index
        else:
            raise AssertionError(f'Node should have 0 or 2 children, not {len(self.children)}')
        names.append(self.name)
        weights.append(self.weight)
        dists.append(self.distance or 0.0)
        parents.append(NO_PARENT)
        children.append((i_left_child, i_right_child))
        return index

    def show(self):
        names = []
        weights = []
        dists = []
        parents = []
        children = []
        self._add_to_arrays(names, weights, dists, parents, children)
        print(children)
        assert len(names) == len(weights) == len(dists) == len(parents) == len(children)
        n = len(names)
        assert n % 2 == 1
        n_leaves = (n+1) // 2

        import numpy as np
        from matplotlib import pyplot as plt
        phi = np.empty((n,), dtype=float)
        r = np.empty((n,), dtype=float)
        color = np.empty((n,), dtype=float)
        leaf_count = 0
        for i in range(n):
            if children[i] == (NO_CHILD, NO_CHILD):
                phi[i] = 2*np.pi * leaf_count / n_leaves
                leaf_count += 1
                color[i] = 0
            else:
                left, right = children[i]
                phi[i] = (phi[left]*weights[left] + phi[right]*weights[right]) / weights[i]
                color[i] = 1
        assert leaf_count == n_leaves, f'{leaf_count}, {n_leaves}'
        r[n-1] = 0.0
        for i in reversed(range(n-1)):
            r[i] = r[parents[i]] + dists[i]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        for i in range(n-1):
            plt.plot((x[i], x[parents[i]]), (y[i], y[parents[i]]), 'c-')
        plt.scatter(x, y, c=color)
        for i in range(n):
            if children[i] == (NO_CHILD, NO_CHILD):
                plt.annotate(names[i], (x[i], y[i]))
        plt.show()
