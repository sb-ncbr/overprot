import sys
import math
import random
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
import json

from libs import lib
from libs.lib import Timing, ProgressBar
from libs.lib_similarity_trees.ghtree import GHTree
from libs.lib_similarity_trees.mtree import MTree
from libs.lib_similarity_trees.nntree import NNTree, _MagicNNTree
from libs.lib_similarity_trees.nearest_pair_keepers import MTNearestPairKeeper


def random_sample_nd(size: int, dim: int = 3) -> List[Tuple[float,...]]:
    result = []
    for i in range(size):
        sample = tuple(random.random() for j in range(dim))
        result.append(sample)
    return result

def dist_nd(sample1: Tuple[float,...], sample2: Tuple[float,...]) -> float:
    sqdist = sum((x-y)**2 for x, y in zip(sample1, sample2))
    return math.sqrt(sqdist)

def dumb_kNN_query(tree, query, k: int, include_query=False):
    candidates = []
    for elem in tree._elements:
        if elem != query or include_query:
            candidates.append((tree.get_distance(query, elem), elem))
    candidates.sort()
    return [elem for dist, elem in candidates[:k]]

def merge_nd(sample1: Tuple[float,...], sample2: Tuple[float,...]) -> Tuple[float,...]:
    return tuple((x+y)/2 for x, y in zip(sample1, sample2))

def test_ght():
    n = 2_000
    keys = [str(i) for i in range(n)]
    samples = random_sample_nd(n, dim=3)
    range_ = 0.2  # for range queries
    k = 1  # for kNN queries
    
    # tree: MTree[str, float] = MTree(distance_function=_dist_nd, leaf_arity=4, internal_arity=4)
    tree = GHTree(dist_nd, leaf_size=8)

    with Timing(f'Adding {n} elements'):
        with ProgressBar(n, title=f'Adding {n} elements') as bar:
            for key, sample in zip(keys, samples):
                tree.insert(key, sample)
                bar.step()
    print(tree_json_subtrees())

    with Timing(f'Querying {n} elements'):
        with ProgressBar(n, title=f'Querying {n} elements') as bar:
            for key, sample in zip(keys, samples):
                result = sorted(tree.range_query(key, range_))
                # check_result = sorted(k for k, s in zip(keys, samples) if _dist_nd(sample, s) <= range_)
                # assert result == check_result
                bar.step()
    print(tree.get_statistics())
            
    with Timing(f'kNN-querying {n} elements'):
        with ProgressBar(n, title=f'kNN-querying {n} elements') as bar:
            for key, sample in zip(keys, samples):
                result = tree.kNN_query(key, k)
                # check_result = dumb_kNN_query(tree, key, k)
                # assert result == check_result
                bar.step()
    print(tree.get_statistics())
    
    with Timing(f'Deleting {n} elements'):
        with ProgressBar(n, title=f'Deleting {n} elements') as bar:
            for key in keys:
                tree.delete(key)
                bar.step()
    print(tree.get_statistics())

    # with Timing(f'kNN-querying and deleting {n} elements'):
    #     with ProgressBar(n, title=f'kNN-querying and deleting {n} elements') as bar:
    #         for key, sample in zip(keys, samples):
    #             if len(tree) >= 2:
    #                 result = tree.kNN_query(key, k)
    #             tree.delete(key)
    #             bar.step()
    # print(tree.get_statistics())

    # dists = np.array([_dist_nd(s, t) for s, t in itertools.combinations(samples, 2)])
    # rsd = dists.std() / dists.mean()
    # print('rsd', rsd)
    # plt.hist(dists)
    # plt.show()
    return

def test_nnt():
    MUTE_PROGRESS_BARS = False
    CHECK_INVARIANTS = False
    
    # n = 10_000
    # keys = range(n)
    # samples = random_sample_nd(n, dim=3)
    # distance_function = dist_nd

    fam = 'cyp_250a'
    dist, keys, _ = lib.read_matrix(f'/home/adam/Workspace/Python/OverProt/data/GuidedAcyclicClustering/{fam}/distance_matrix.tsv')
    samples = keys
    n = len(keys)
    key_index = {key: i for i, key in enumerate(keys)}
    distance_function = lambda i, j: dist[key_index[i], key_index[j]]
    random.shuffle(keys)
    
    shuffled_key_samples = list(zip(keys, samples))
    random.shuffle(shuffled_key_samples)

    tree = NNTree(distance_function, with_nearest_pair_queue=True)
    # tree = _MagicNNTree(distance_function, with_nearest_pair_queue=True)
    # tree = MTNearestPairKeeper(distance_function)

    with Timing(f'Adding {n} elements'):
        with ProgressBar(n, title=f'Adding {n} elements', mute=MUTE_PROGRESS_BARS) as bar:
            for key, sample in zip(keys, samples):
                # print('Adding', key)
                tree.insert(key, sample)
                if CHECK_INVARIANTS: tree._check_invariants()
                bar.step()
        # print(tree.get_statistics())
        # with open('tmp/nn-tree.json', 'w') as w:
        #     json.dump(tree._tree.json(), w, indent=4)

        # with Timing(f'kNN-querying {n} elements'):
        #     with ProgressBar(n, title=f'kNN-querying {n} elements') as bar:
        #         for key, sample in zip(keys, samples):
        #             result = tree.kNN_query(key, k, include_query=False)
        #             # result2 = tree2.NN_query(key)
        #             # result2 = tree2.kNN_query_bottom_up(key, k, include_query=False)
        #             # result_pp = tree2.kNN_query_classical(key, k, include_query=False, pivot_prox=True)
        #             # check_result = dumb_kNN_query(tree2, key, k, include_query=False)
        #             # assert result == check_result, f'\n{result}\n{check_result}'
        #             # assert result_pp == result, f'\n{result_pp}\n{result}'
        #             # if result != result2:
        #             #     print(*((r, tree.get_distance(key, r)) for r in result))
        #             #     print(*((r, tree.get_distance(key, r)) for r in result2))
        #             #     raise AssertionError
        #             bar.step()
        # print(tree.get_statistics())

        # shuffled_key_samples.reverse()
        # with Timing(f'Deleting {n} elements'):
        #     with ProgressBar(n, title=f'Deleting {n} elements', mute=MUTE_PROGRESS_BARS) as bar:
        #         for key, sample in shuffled_key_samples:
        #             # print('\nDeleting', key)
        #             tree.delete(key)
        #             if CHECK_INVARIANTS: tree._check_invariants()
        #             bar.step()
        # print(tree.get_statistics())
    
    # with Timing(f'Popping {n} elements as nearest pairs'):
        with ProgressBar(n//2, title=f'Popping {n} elements as nearest pairs', mute=MUTE_PROGRESS_BARS) as bar:
            while len(tree) >= 2:
                p = tree.pop_nearest_pair()
                if CHECK_INVARIANTS: tree._check_invariants()
                bar.step()

    print(tree.get_statistics())


def test_nearest_pair_keeper():
    n = 1_000
    keys = [str(i) for i in range(n)]
    samples = random_sample_nd(n, dim=3)
    sample_dict = dict(zip(keys, samples))

    with Timing(f'Fill GHTNearestPairKeeper with {n} elements'):
        keeper = GHTNearestPairKeeper(dist_nd, sample_dict.items())
    print(keeper._tree.get_statistics())

    with Timing(f'Empty GHTNearestPairKeeper with {n} elements'):
        while len(keeper) >= 2:
            i, j, dist = keeper.pop_nearest_pair()
            # new = f'{i}_{j}'
            # new_sample = merge_nd(sample_dict[i], sample_dict[j])
            # sample_dict[new] = new_sample
            # keeper.insert(new, new_sample)
    # print(keeper._tree)
    print(len(keeper))
    print(keeper._tree.get_statistics())
    # TODO why is distance_cache not empty ?!

if __name__ == '__main__':
    # test_ght()
    # test_nearest_pair_keeper()
    test_nnt()
    # with ProgressBar(1000) as bar:
    #     for _ in range(1000):
    #         test_mdt()
    #         bar.step()
    