'''Library of functions related to clustering'''


from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, List, Sequence

from . import lib


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



def children_to_newick_aux(children, node_index, distances=defaultdict(lambda: 1), node_names=None):
    if children[node_index, 0] < 0:  # it is a leaf
        return (str(node_index) if node_names is None else node_names[node_index])
    else:
        dist = distances[node_index]
        subtree1 = children_to_newick_aux(children, children[node_index, 0], distances=distances, node_names=node_names)
        subtree2 = children_to_newick_aux(children, children[node_index, 1], distances=distances, node_names=node_names)
        return f'({subtree1}:{dist},{subtree2}:{dist})'

def children_to_newick(children, node_indices=[-1], distances=None, node_names=None):
    if distances is None:
        distances = defaultdict(lambda: 1)
    subtrees = [children_to_newick_aux(children, idx, distances=distances, node_names=node_names) for idx in node_indices]
    if len(subtrees) == 1:
        return subtrees[0]+ ';'
    else:
        return '(' + ','.join(subtrees) + ');'

def children_to_list(children, root=None, node_names=None):
    if root is None:
        root = children.shape[0] - 1
    left, right = children[root,:]
    if left == -1:  # root is leaf
        return [node_names[root]] if node_names is not None else [root]
    else:  # root is inner leaf
        return children_to_list(children, root=left, node_names=node_names) + children_to_list(children, root=right, node_names=node_names)

def children_to_newick_limited(children, data, max_leaves=-1):
    power = 1
    n = children.shape[0]
    max_splits = min(max_leaves-1, n) if max_leaves>=1 else n
    # nodes = [data.snapshots[i]+'_'+data.tunnels[i] for i in range(n+1)]
    nodes = [str(i) for i in range(n+1)]
    counts = [1 for i in range(n+1)]
    class_counts = [{ data.class_names[data.y[i]]: 1} for i in range(n+1)]
    indices = [[i] for i in range(n+1)]
    residual_ss = [0 for i in range(n+1)]
    for i in range(n):
        child0 = children[i,0]
        child1 = children[i,1]
        nodes.append('(' + nodes[child0] + ',' + nodes[child1] + ')')
        counts.append(counts[child0] + counts[child1])
        class_counts.append(sum_dict(class_counts[child0], class_counts[child1]))
        indices.append(indices[child0] + indices[child1])
        residual_ss.append(residual_ss[-1] + sum_sq_deficit(data.X, indices[child0], indices[child1], power))
    cut_nodes = []
    start = len(nodes)-max_splits
    print('n:', n)
    print('len(nodes):', len(nodes))
    print('start:', start)
    for i in range(start, len(nodes)):
        names = []
        childs = children[i-n-1]
        for j in [0, 1]:
            child = childs[j]
            if child < start:
                #names.append(str(counts[child]))
                clc = class_counts[child]
                clc = [(clc[c], c) for c in clc]
                name = '[' + '  '.join(str(t[0])+'.'+t[1] for t in sorted(clc, reverse=True)) + ']' #str(class_counts[child])
                #name = '[' + '  '.join(str(clc[c])+'.'+c for c in sorted(clc, lambda )) + ']' #str(class_counts[child])
                #name = str(counts[child])
                names.append(name)
            else:
                names.append(cut_nodes[child - start])
        ss_def = sum_sq_deficit(data.X, indices[childs[0]], indices[childs[1]], power)
        depth0 = (residual_ss[i] - residual_ss[childs[0]])
        depth1 = (residual_ss[i] - residual_ss[childs[1]])
        cut_nodes.append('(' + names[0] + ':' + str(depth0) +  ',' + names[1] + ':' + str(depth1) + ')')
        #cut_nodes.append('(' + names[0] + ':' + str(ss_def) +  ',' + names[1] + ':' + str(ss_def) + ')')
    print('cut nodes:', len(cut_nodes))
    plt.figure(figsize=(10,10))
    #plt.plot(residual_ss[-1:-50:-1])
    plt.plot([residual_ss[-1-i]-residual_ss[-2-i] for i in range(20)], '.-')
    plt.show()
    return cut_nodes[-1] + ';'

def complete_linkage_aggregate_function(value1, size1, value2, size2):
    return max(value1, value2)

def average_linkage_aggregate_function(value1, size1, value2, size2):
    return (value1*size1 + value2*size2) / (size1+size2)

def average_linkage_aggregate_function_with_size_adhesion(value1, size1, value2, size2, adhesion_strength=1):
    return (value1 * size1**(1+adhesion_strength) + value2 * size2**(1+adhesion_strength)) / (size1 + size2)**(1+adhesion_strength)

