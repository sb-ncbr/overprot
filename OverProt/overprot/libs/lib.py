'''Library of general-purpose functions'''


import os
from os import path
import glob
import sys
import shutil
import re
from datetime import datetime
import heapq
import numpy as np
import itertools
import subprocess
from contextlib import contextmanager, suppress
import multiprocessing
import configparser
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Iterable, Iterator, TypeVar, Union, Optional, NamedTuple, Generic, Callable, TextIO, Any, Literal, Sequence, Mapping, Final, Type, get_origin, get_args, get_type_hints

K = TypeVar('K')
V = TypeVar('V')


COMMENT_SYMBOL = '#'
RE_DTYPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*dtype\s*=\s*(\w+)\s*$')
RE_SHAPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*shape\s*=\s*(\w+)\s*,\s*(\w+)\s*$')

def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()  # necessary when importing pymol because it somehow fucks up stdout

def log_debug(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()  # necessary when importing pymol because it somehow fucks up stdout

def run_command(*args, stdin: Optional[str] = None, stdout: Optional[str] = None, stderr: Optional[str] = None, 
                appendout: bool = False, appenderr: bool = False) -> int:
    out_mode = 'a' if appendout else 'w'
    err_mode = 'a' if appenderr else 'w'
    with maybe_open(stdin, 'r', default=sys.stdin) as stdin_handle:
        with maybe_open(stdout, out_mode, default=sys.stdout) as stdout_handle: 
            with maybe_open(stderr, err_mode, default=sys.stderr) as stderr_handle:
                # print(' '.join(map(str, args)))
                process = subprocess.run(list(map(str, args)), check=True, stdin=stdin_handle, stdout=stdout_handle, stderr=stderr_handle)
    return process.returncode

def run_dotnet(dll: str, *args, **run_command_kwargs):
    if not os.path.isfile(dll):  # dotnet returns random exit code, if the DLL is not found ¯\_(ツ)_/¯
        raise FileNotFoundError(dll)
    run_command('dotnet', dll, *args, **run_command_kwargs)

def try_remove_file(filename: str) -> bool:
    '''Try to remove a file ($ rm -f filename). Return True if the file has been successfully removed, False otherwise.'''
    try:
        os.remove(filename)
        return True
    except OSError:
        return False

def clear_file(filename: str) -> None:
    '''Remove all text from a file (or create empty file if does not exist).'''
    with open(filename, 'w'): 
        pass

def unique(iterable):
    '''Remove duplicates. Usable when set() is not usable because of non-hashable type'''
    result = []
    for elem in iterable:
        if elem not in result:
            result.append(elem)
    return result

def are_unique(iterable):
    '''Decide whether an iterable contains duplicates.'''
    seen = set()
    for item in iterable:
        if item in seen:
            return False
        else:
            seen.add(item)
    return True
    # c = collections.Counter(iterable)
    # return max(c.values()) == 1

def single(iterable: Iterable[V]) -> V:
    '''Return the single element of the iterable, or raise ValueError if len(iterable) != 1'''
    iterator = iter(iterable)
    try:
        value = next(iterator)
        try:
            _ = next(iterator)
            raise ValueError('Iterable contains more than one element)')
        except StopIteration:
            return value
    except StopIteration:
        raise ValueError('Iterable contains no elements')

def not_none(*values: Optional[V]) -> V:
    '''Return first non-None value, or raise ValueError if all values are None.'''
    for value in values:
        if value is not None:
            return value
    raise ValueError

def str_join(sep: str, elements: List[V], three_dots_after: int = -1):
    """Do the same as sep.join(elements), but automatically convert elements to strings and include at most three_dots_after elements. Put '...' at the end if there are more elements"""
    if three_dots_after == -1 or len(elements) > three_dots_after:
        return sep.join(str(elem) for  elem in elements[:three_dots_after]) + '...'
    else:
        return sep.join(str(elem) for elem in elements)

def insert_after(dictionary: Dict[K, V], after_what: K, new_key_value_pairs: Iterable[Tuple[K, V]]) -> None:
    key_value_pairs = list(dictionary.items())
    dictionary.clear()
    for key, value in key_value_pairs:
        dictionary[key] = value
        if key == after_what:
            for k, v in new_key_value_pairs:
                dictionary[k] = v

def find_indices_where(iterable, predicate):
    return [i for i, elem in enumerate(iterable) if predicate(elem)]

def get_offsets(iterable, key=lambda x: x):  # Returns starting indices of regions with the same value in iterable and puts extra len(iterable) at the end
    keys = []
    offsets = []
    current_key = None
    length = 0
    for i, elem in enumerate(iterable):
        new_key = key(elem)
        length += 1
        if i == 0 or new_key != current_key:
            keys.append(new_key)
            offsets.append(i)
            current_key = new_key
    offsets.append(length)
    return keys, offsets

def invert_offsets(offsets):  # Produces inverse mapping for offsets, e.g. [0, 3, 5, 10] => [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    n = len(offsets)-1
    result = []
    for i in range(n):
        fro = offsets[i]
        to = offsets[i+1]
        for j in range(fro, to):
            result.append(i)
    return result

def read_matrix(filename: 'FilePath', sep='\t', dtype=None) -> Tuple[np.ndarray, List[str], List[str]]:
    with filename.open() as f:
        col_names = None
        row_names = []
        values = []
        shape = None
        matrix = None
        row_count = 0
        for line in iter(f.readline, ''):
            if line.strip('\r\n') == '':
                # ignore empty line
                pass
            elif line[0] == COMMENT_SYMBOL:
                # read comment
                if dtype is None and RE_DTYPE.match(line):
                    dtype = RE_DTYPE.sub('\\1', line)
                elif RE_SHAPE.match(line):
                    nrows = int(RE_SHAPE.sub('\\1', line))
                    ncols = int(RE_SHAPE.sub('\\2', line))
                    shape = (nrows, ncols)
            elif line[0] == sep:
                # read column names
                _, *col_names = line.strip('\r\n').split(sep)
                col_names = [name for name in col_names if name != '']
                # initialize matrix if the shape is known
                if shape is not None:
                    matrix = np.zeros(shape, dtype=dtype) if dtype is not None else np.zeros(shape)
                    # log('Init. matrix')
            else:
                # read a row name + values
                row_name, *vals = line.strip('\r\n').split(sep)
                row_names.append(row_name)
                row_values = [float(x) for x in vals if x!='']
                if matrix is not None:  # initialized matrix => write directly to matrix
                    matrix[row_count,:] = row_values
                    row_count += 1
                else:  # uninitialized matrix => write to a list of lists
                    values.append([float(x) for x in vals if x!=''])
        if col_names is None:
            print(col_names, row_names, values, matrix)
            raise Exception(f'{filename} contains no column names')
    if matrix is None:
        matrix = np.array(values, dtype=dtype) if dtype is not None else np.array(values)
    else:
        pass  # the matrix has already been initialized and filled
    return matrix, row_names, col_names

def print_matrix(matrix, filename, row_names=None, col_names=None, sep='\t'):
    m, n = matrix.shape
    if row_names is None:
        r_digits = len(str(m-1))
        row_name_template = f'r{{:{r_digits}}}'
        row_names = [row_name_template.format(i) for i in range(m)]
    if col_names is None:
        c_digits = len(str(n-1))
        col_name_template = f'c{{:{c_digits}}}'
        col_names = [col_name_template.format(i) for i in range(n)]
    if matrix.dtype==bool:
        str_ = lambda x: '1' if x else '0'
    elif matrix.dtype==float:
        str_ = '{:g}'.format
    else:
        str_ = str
    with open(filename, 'w') as g:
        g.write(COMMENT_SYMBOL + 'dtype='+str(matrix.dtype) + '\n')
        g.write(COMMENT_SYMBOL + 'shape='+str(m)+','+str(n) + '\n')
        if row_names is not None and col_names is not None:
            g.write(sep)
        if col_names is not None:
            g.write(sep.join(col_names) + '\n')
        for i in range(m):
            if row_names is not None:
                g.write(row_names[i] + sep)
            g.write(sep.join((str_(x) for x in matrix[i,:])) + '\n')

def replace(matrix, orig_value, new_value):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            if matrix[i,j]==orig_value:
                matrix[i,j] = new_value

def submatrix_int_indexing(matrix, *indices, put=None):
    '''
    Return a submatrix if put is None, or replace submatrix in the matrix by put if put is not None.
    Use list-of-locations indexing, i.e. indices for each dimension are a sequence of int.
    '''
    try:
        matrix.shape
    except AttributeError:  # matrix is not numpy array
        matrix = np.array(matrix)
    indices = [ idx if hasattr(idx, 'dtype') else np.empty((0,), dtype=int) if len(idx) == 0 else np.array(idx) for idx in indices ]
    if not all( idx.dtype == int for idx in indices ):
        raise TypeError('Indices for each dimension must be a sequence of int')
    shape = list(matrix.shape)
    shape[:len(indices)] = ( len(idx) for idx in indices )
    meshes = [None] * len(indices)
    if len(indices) < 2:
        meshes = indices
    else:
        meshes = [None] * len(indices)
        meshes[1], meshes[0], *meshes[2:] = ( mesh.reshape(-1) for mesh in np.meshgrid(indices[1], indices[0], *indices[2:], copy=False) )
    # cmesh, rmesh = ( mesh.reshape(-1) for mesh in np.meshgrid(col_indices, row_indices, copy=False) )
    if put is None:
        return matrix[tuple(meshes)].reshape(shape)
    else:
        matrix[tuple(meshes)] = put.reshape(-1)

def submatrix_bool_indexing(matrix, *indices, put=None):
    '''
    Return a submatrix if put is None, or replace submatrix in the matrix by put if put is not None.
    Use boolean indexing, i.e. indices for dimension i are a sequence of bool with length matrix.shape[i].
    '''
    indices = [ idx if hasattr(idx, 'dtype') else np.empty((0,), dtype=bool) if len(idx) == 0 else np.array(idx) for idx in indices ]
    if not all( idx.dtype == bool for idx in indices ):
        raise TypeError('Indices for dimension i must be a sequence of bool with length matrix.shape[i]')
    shape = list(matrix.shape)
    shape[:len(indices)] = ( idx.sum() for idx in indices )
    selected = np.array(True)
    for idx in indices:
        selected = np.logical_and(np.expand_dims(selected, -1), idx)
    # selected = each_to_each(np.logical_and, row_indices, col_indices)
    if put is None:
        return matrix[selected].reshape(shape)
    else:
        matrix[selected] = put.reshape(-1)

def test_submatrix():
    A = np.arange(1000).reshape((10,10,10))
    B = np.arange(10)
    submatrix_int_indexing(A, [0,1,2], [5], [7,9], put=np.array(0))
    print(A)

def each_to_each(function, vector, vector2=None):
    '''Creates numpy matrix M, where M[i,j] = function(vector[i], vector2[j]). 
    function: numpy broadcastable function
    vector: numpy 1D array
    vector2: numpy 1D array; if not provided then vector2 == vector '''
    if vector2 is None:
        vector2 = vector
    return function(vector.reshape((-1, 1)), vector2.reshape((1, -1)))

def create_lookup(array):
    result = {}
    for i, elem in enumerate(array):
        result[elem] = i
    return result

def sort_dag(vertices, does_precede):
    todo = list(vertices)
    done = []
    while len(todo) > 0:
        mins = [i for i in todo if not any(i!=j and does_precede(j, i) for j in todo)]
        if len(mins)==0:
            raise Exception('Cyclic graph.')
        done.extend(mins)
        todo = [i for i in todo if i not in mins]
        # if len(mins) > 1:
        # 	log('Sort DAG: uncomparable vertices:', *mins)
    return done

class LabeledEdge(NamedTuple):
    invertex: int
    outvertex: int
    label: int

def left_subdag_tipsets(n_vertices: int, edges: List[Tuple[int,int]], precedence_matrix: Optional[np.ndarray]=None) -> Tuple[List[Tuple[int,...]], List[LabeledEdge]]:
    '''Return the list of tipsets of all left subdags of dag H and all successor left subdag pairs (as index tuples).
    G is a left subdag of dag H iff:
        foreach u-v path in H. if v in G then u in G
    Tipset of dag G is the minimal vertex subset S such that:
        foreach u in G. exists v in S. exists path u-v
    G, G' are a successor left subdag pair of dag H iff:
        G, G' are left subdags of H and G can be obtained by removing one vertex from G'.
    '''
    vertices = range(n_vertices)
    if precedence_matrix is None:
        with Timing('precedence_matrix', mute=True):
            precedence_matrix = precedence_matrix_from_edges_dumb(n_vertices, edges)
    with Timing('left_subdag_tipsets', mute=True):
        nontips = {u for (u, v) in edges}
        tipset = tuple(v for v in vertices if v not in nontips)
        active_tipsets = {tipset: 0}
        tipset_counter = 1
        all_tipsets: List[Tuple[int,...]] = []
        tipset_edges = []
        while len(active_tipsets) > 0:
            all_tipsets.extend(active_tipsets.keys())
            new_tipsets = {}
            for tipset, tipset_id in active_tipsets.items():
                incoming_edges = [(u,v) for (u,v) in edges if v in tipset]
                for tip in tipset:
                    potential_subtips = [u for (u,v) in incoming_edges if v == tip]
                    subtips = []
                    for u in potential_subtips:
                        if any(precedence_matrix[u,v] and v != tip for v in tipset):  # TODO this will cause problem if there is subgraph [A->B, B->C, A->C]
                            continue
                        subtips.append(u)
                    new_tipset = tuple(sorted(subtips + [t for t in tipset if t != tip]))
                    if new_tipset not in new_tipsets:
                        new_tipset_id = tipset_counter
                        new_tipsets[new_tipset] = new_tipset_id
                        tipset_counter += 1
                    else:
                        new_tipset_id = new_tipsets[new_tipset]
                    tipset_edges.append(LabeledEdge(new_tipset_id, tipset_id, tip))
            active_tipsets = new_tipsets
    n_tipsets = len(all_tipsets)
    all_tipsets = all_tipsets[::-1]
    tipset_edges = [LabeledEdge(n_tipsets-1-i, n_tipsets-1-j, tip) for (i, j, tip) in tipset_edges]
    assert are_edge_labels_unique_within_each_oriented_path(tipset_edges)
    return all_tipsets, tipset_edges

def are_edge_labels_unique_within_each_oriented_path(edges: List[LabeledEdge]):
    edge_starts = {u for u, v, lab in edges}
    edge_ends = {v for u, v, lab in edges}
    vertices = list(edge_starts | edge_ends)
    inflowing_labels: Dict[int, Set[int]] = {v: set() for v in vertices}
    while True:
        changed = False
        for u, v, lab in edges:
            if lab not in inflowing_labels[v]:
                changed = True
                inflowing_labels[v].add(lab)
            if lab in inflowing_labels[u]:
                return False
        if not changed:
            return True

def precedence_matrix_from_edges_dumb(n_vertices, edges):
    # n_vertices = len(vertices)
    # assert all(vertices[i] < vertices[i+1] for i in range(n_vertices-1))
    vertices = range(n_vertices)
    precedence = np.eye(n_vertices, dtype=bool)
    while True:
        changed = False
        for u, v in edges:
            for w in vertices:
                if not precedence[u, w] and precedence[v, w]:
                    changed = True
                    precedence[u, w] = True
        if not changed:
            break
    for v in vertices:
        precedence[v, v] = False
    return precedence

def test_left_subdag_tipsets():
    # n_vertices = 10
    # edges = [(i, i+1) for i in range(n_vertices-1)]
    # edges = [(0,1), (1,2), (2,3), (3,4), (1,5), (1,6), (4,7), (5,7), (7,8), (6,8), (8,9)]
    n_vertices = 13
    edges = [(0,1), (1,2), (2,3), (3,4), (4,10), (10,11), (11,12), (1,5), (5,6), (6,4), (6,7), (7,8), (8,9), (9,11)]
    tipsets, tipset_edges = left_subdag_tipsets(n_vertices, edges)
    print(tipsets)
    s = set()
    for (i,j,tip) in tipset_edges:
        s.add((tipsets[i], tipsets[j], tip))
    for g, h, t in sorted(s):
        print(f'{g} + {t} -> {h}')
    print(len(tipsets))

# def connected_components(edges, vertex_sort_key=(lambda x: x)):
#     vertices = set()
#     for u, v, *_ in edges:
#         vertices.add(u)
#         vertices.add(v)
#     components = [ [v] for v in vertices ]
#     for u, v, *_ in edges:
#         comp_u = next(comp for comp in components if u in comp)
#         comp_v = next(comp for comp in components if v in comp)
#         if comp_u != comp_v:
#             components.remove(comp_u)
#             components.remove(comp_v)
#             components.append(comp_u + comp_v)
#     for comp in components:
#         comp.sort(key=vertex_sort_key)
#     components.sort(key=lambda comp: vertex_sort_key(comp[0]))
#     return components

def safe_mean(X, axis=None, fallback_value=0.0):
    '''Same as numpy.mean, but on empty slices returns fallback_value instead of NaN; does not throw any warning.'''
    if not hasattr(X, 'size'):
        X = np.array(X)
    if X.size > 0:
        return np.mean(X, axis=axis)
    elif axis is None:
        return fallback_value
    else:
        new_shape = list(X.shape)
        new_shape.pop(axis)
        return np.full(tuple(new_shape), fallback_value)

def safe_corrcoef(X):
    '''Same as numpy.corrcoef, but if Var(x)==0 then Corr(x,y)==0; does not throw division-by-zero warning. Variables in rows, observations in columns.'''
    m, n = X.shape
    regular_rows = np.nonzero(np.var(X, axis=1) > 0)[0]
    reg_corr = np.corrcoef(X[regular_rows, :])
    corr = np.zeros((m, m), dtype=float)
    submatrix_int_indexing(corr, regular_rows, regular_rows, put=reg_corr)
    return corr

def integer_suffix(string, empty_means_zero=True):
    for i in range(len(string)):
        try:
            return int(string[i:])
        except ValueError:
            pass
    if empty_means_zero:
        return 0
    else:
        raise ValueError(f'String {string} contains no integer suffix')


def matrix_from_function(shape: Tuple[int, ...], function: Callable, dtype=float, symmetric: bool = False, diag_value: Optional[Any] = None) -> np.ndarray:
    '''Create matrix of given shape, where matrix[i, j, ...] == function(i, j, ...).
    If symmetric==True, then it is assumed that matrix[..., i, j] == matrix[..., j, i].
    If diag_value is not None, then it is assumed that matrix[..., i, i] == diag_value.'''
    matrix = np.empty(shape, dtype=dtype)
    if symmetric:
        if diag_value is None:
            for idcs in itertools.product(*(range(n) for n in shape)):
                matrix[idcs] = function(*idcs) if idcs[-2] < idcs[-1] else matrix[idcs[:-2]+idcs[-1:-3:-1]]
        else:
            for idcs in itertools.product(*(range(n) for n in shape)):
                matrix[idcs] = function(*idcs) if idcs[-2] < idcs[-1] else matrix[idcs[:-2]+idcs[-1:-3:-1]] if idcs[-2] > idcs[-1] else diag_value
    else:
        if diag_value is None:
            for idcs in itertools.product(*(range(n) for n in shape)):
                matrix[idcs] = function(*idcs)
        else:
            for idcs in itertools.product(*(range(n) for n in shape)):
                matrix[idcs] = function(*idcs) if idcs[-2] != idcs[-1] else diag_value
    return matrix

@contextmanager
def maybe_open(filename: Optional[str], *args, default=None, **kwargs):
    if filename is not None:
        f = open(filename, *args, **kwargs)
        has_opened = True
    else:
        f = default
        if f is not None:
            f.flush()
        has_opened = False
    try:
        yield f
    except Exception:
        raise
    finally:
        if has_opened:
            f.close()


class ProgressBar_Old:
    def __init__(self, n_steps, width=100, title='', writer=sys.stdout):
        self.n_steps = n_steps # expected number of steps
        self.width = width
        self.title = (' '+title+' ')[0:min(len(title)+2, width)]
        self.writer = writer
        self.done = 0 # number of completed steps
        self.shown = 0 # number of shown symbols
    def start(self):
        self.writer.write('|' + self.title + '_'*(self.width-len(self.title)) + '|\n')
        self.writer.write('|')
        self.writer.flush()
        return self
    def step(self, n_steps=1):
        self.done = min(self.done + n_steps, self.n_steps)
        new_shown = int(self.width * self.done / self.n_steps if self.n_steps > 0 else self.width)
        self.writer.write('*' * (new_shown-self.shown))
        self.writer.flush()
        self.shown = new_shown
    def finalize(self):
        self.step(self.n_steps - self.done)
        self.writer.write('|\n')
        self.writer.flush()


class ProgressBar:
    DONE_SYMBOL = '█'
    TODO_SYMBOL = '-'

    def __init__(self, n_steps: int, *, width: Optional[int] = None, 
                 title: str = '', prefix: Optional[str] = None, suffix: Optional[str] = None, 
                 writer: Union[TextIO, Literal['stdout', 'stderr']] = 'stdout', mute: bool = False):
        self.n_steps = n_steps # expected number of steps
        self.prefix = prefix + ' ' if prefix is not None else ''
        self.suffix = ' ' + suffix if suffix is not None else ''
        self.width = width if width is not None else shutil.get_terminal_size().columns
        self.width -= len(self.prefix) + len(self.suffix) + 10  # self.width -= len(prefix) + len(suffix) + 8
        self.title = (' '+title+' ')[0:min(len(title)+2, self.width)]
        self.writer = sys.stdout if writer == 'stdout' else sys.stderr if writer == 'stderr' else writer  # writer if writer is not None else sys.stdout
        self.done = 0 # number of completed steps
        self.shown = 0 # number of shown symbols
        self.mute = mute

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finalize(completed = exc_type is None)

    def start(self):
        if not self.mute:
            self.writer.write(' ' * len(self.prefix))
            self.writer.write('┌' + self.title + '─'*(self.width-len(self.title)) + '┐\n')
            self.writer.flush()
            self.step(0, force=True)
        return self

    def step(self, n_steps=1, force=False):
        if not self.mute:
            if n_steps == 0 and not force:
                return
            self.done = min(self.done + n_steps, self.n_steps)
            try:
                progress = self.done / self.n_steps
            except ZeroDivisionError:
                progress = 1.0
            new_shown = int(self.width * progress)
            if new_shown != self.shown or force:
                self.writer.write(f'\r{self.prefix}└')
                self.writer.write(self.DONE_SYMBOL * new_shown + self.TODO_SYMBOL * (self.width - new_shown))
                self.writer.write(f'┘ {int(100*progress):>3}%{self.suffix} ')
                self.writer.flush()
                self.shown = new_shown  

    def finalize(self, completed=True):
        if not self.mute:
            if completed:
                self.step(self.n_steps - self.done)
            self.writer.write('\n')
            self.writer.flush()
    

class Counter(Generic[K]):
    def __init__(self):
        self.dict = {}
    def add(self, elem: K) -> None:
        if elem in self.dict:
            self.dict[elem] += 1
        else:
            self.dict[elem] = 1
    def count(self, elem: K) -> int:
        if elem in self.dict:
            return self.dict[elem]
        else:
            return 0
    def all_counts(self) -> Dict[K, int]:
        return self.dict


class PriorityQueue(Generic[K, V]):
    def __init__(self, keys_elements: Iterable[Tuple[K, V]] = None):
        if keys_elements is None:
            keys_elements = []
        self.heap = list(( (key, i, elem) for (i, (key, elem)) in enumerate(keys_elements) )) 
        heapq.heapify(self.heap)
        self.seq = len(self.heap)
    def is_empty(self) -> bool:
        return len(self.heap) == 0
    def add(self, key: K, elem: V) -> None:
        heapq.heappush(self.heap, (key, self.seq, elem))
        self.seq += 1
    def get_min(self) -> Optional[Tuple[K, V]]:
        '''Return the minimal element. Return None if the queue is empty.'''
        if not self.is_empty():
            key, i, elem = self.heap[0]
            return key, elem
        else:
            return None
    def pop_min(self) -> Optional[Tuple[K, V]]:
        '''Return and discard the minimal element. Return None if the queue is empty.'''
        if not self.is_empty():
            key, i, elem = heapq.heappop(self.heap)
            return key, elem
        else:
            return None
    def pop_min_which(self, predicate: Callable[[V], bool]) -> Optional[Tuple[K, V]]:
        '''Return and remove the minimal element fullfilling predicate, and remove all the preceding elements. Return None and empty the queue, if no such element.'''
        while not self.is_empty():
            key, i, elem = heapq.heappop(self.heap)
            if predicate(elem):
                return key, elem
        return None
    def __len__(self) -> int:
        return len(self.heap)


class NumpyMinHeap:
    def __init__(self, value_shape, value_type, capacity, key_type=float, keys_values=None):
        self.n = 0
        self.keys = np.empty(capacity + 1, dtype=key_type)
        self.values = np.empty((capacity + 1, *value_shape), dtype=value_type)
        if keys_values is not None:
            for key, value in keys_values:
                self.add(key, value)
    def is_empty(self):
        return self.n == 0
    def add(self, key, value):
        if self.n == self.values.shape[0] - 1:
            raise Exception('Queue overflow.')
        self.n += 1
        index = self.n - 1
        while True:
            parent = (index - 1) // 2
            if parent >= 0 and self.keys[parent] > key:
                self.keys[index] = self.keys[parent]
                self.values[index] = self.values[parent]
                index = parent
            else: 
                self.keys[index] = key
                self.values[index] = value
                break
    def pop_min(self):
        '''Discard the minimal element and return its key and view of the value. Return None if the queue is empty.'''
        min_key = self.keys[0]
        self.values[-1] = self.values[0]
        self.n -= 1
        index = 0
        the_key = self.keys[self.n]
        while True:
            left = 2 * index + 1
            right = left + 1
            if right < self.n and self.keys[right] < self.keys[left]:
                if self.keys[right] < the_key:
                    # shake right!
                    self.keys[index] = self.keys[right]
                    self.values[index] = self.values[right]
                    index = right
                else:
                    # finish
                    self.keys[index] = the_key
                    self.values[index] = self.values[self.n]
                    break
            elif left < self.n and self.keys[left] < the_key:
                # shake left!
                self.keys[index] = self.keys[left]
                self.values[index] = self.values[left]
                index = left
            else:
                # finish
                self.keys[index] = the_key
                self.values[index] = self.values[self.n]
                break
        return min_key, self.values[-1]
    def pop_min_which(self, predicate):
        while not self.is_empty():
            key, value = self.pop_min()
            if predicate(value):
                return key, value
        return None


class Timing:
    def __init__(self, name: Optional[str] = None, file: Union[TextIO, Literal['stdout', 'stderr']] = 'stderr', mute=False):
        self.name = name
        self.file = sys.stdout if file == 'stdout' else sys.stderr if file == 'stderr' else file
        self.mute = mute
        self.time = None
    def __enter__(self):
        self.t0 = datetime.now()
        return self
    def __exit__(self, *args):
        dt = datetime.now() - self.t0
        self.time = dt
        if not self.mute:
            if self.name is not None:
                message = f'Timing: {self.name}: {dt}'
            else:
                message = f'Timing: {dt}'
            print(message, file=self.file)



class Tee:
    def __init__(self, *outputs: TextIO):
        self.outputs  = outputs
    def write(self, *args, **kwargs):
        for out in self.outputs:
            out.write(*args, **kwargs)
    def flush(self, *args, **kwargs):
        for out in self.outputs:
            out.flush(*args, **kwargs)



_ConfigOptionValue = Union[str, int, float, bool, List[str], List[int], List[float], List[bool]]

class ConfigSection(object):
    '''Represents one section of configuration like in .ini file.
    Subclasses of ConfigSection can declare instance variables corresponding to individual options in the section, these should be of one of the types given by _ConfigOptionValue.
    Instance variables with prefix _ are ignored.
    '''

    __ALLOWED_TYPES: Final = get_args(_ConfigOptionValue)
    __DEFAULT_TYPE: Final = str
    __option_types: Dict[str, type]

    def __init__(self):
        cls = type(self)
        self.__option_types = {option: typ for option, typ in get_type_hints(cls).items() if not option.startswith('_')}
        for option, typ in self.__option_types.items():
            allowed_types = ', '.join(str(t) for t in self.__ALLOWED_TYPES) + ' or Literal of strings'
            assert typ in self.__ALLOWED_TYPES or self._is_string_literal(typ), f'Type {typ} is not allowed for options in Config ({cls.__name__}.{option}). Allowed types: {allowed_types}'
            if option in vars(cls):
                value = vars(cls)[option]
            elif get_origin(typ) is None:  # non-generic
                value = typ()
            elif get_origin(typ) is list:
                value = list()
            elif get_origin(typ) is Literal:
                value = get_args(typ)[0]
            else:
                raise NotImplementedError(typ)
            self.__setattr__(option, value)
    
    def __str__(self) -> str:
        lines = []
        for option in vars(self):
            if not option.startswith('_'):
                value = self.__getattribute__(option)
                if isinstance(value, str):
                    value = value.replace('\n', '\n\t')
                elif isinstance(value, list):
                    value = '\n\t'.join(str(elem) for elem in value)
                lines.append(f'{option} = {value}')
        return '\n'.join(lines)

    def __repr__(self) -> str:
        opts = []
        for option in vars(self):
            if not option.startswith('_'):
                value = self.__getattribute__(option)
                opts.append(f'{option}={repr(value)}')
        options = ', '.join(opts)
        return f'{type(self).__name__}({options})'

    def _set_options(self, parser: configparser.ConfigParser, section: str, allow_extra: bool = False, allow_missing: bool = False, filename: str = '???') -> None:
        options = parser[section]

        if not allow_extra:
            for option in options:
                assert option in self.__option_types, f'Extra option {option} in section [{section}] in file {filename}'
        else:
            for option in options:
                if option not in self.__option_types:
                    self.__option_types[option] = self.__DEFAULT_TYPE
                    self.__setattr__(option, self.__DEFAULT_TYPE())

        if not allow_missing:
            for option in self.__option_types:
                assert option in options, f'Missing option {option} in section [{section}] in file {filename}'
            
        for option in options:
            option_type = self.__option_types.get(option, self.__DEFAULT_TYPE)
            typed_value: _ConfigOptionValue
            if option_type == str:
                typed_value = parser.get(section, option)
            elif option_type == int:
                try:
                    typed_value = parser.getint(section, option)
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be an integer.')
            elif option_type == float:
                try:
                    typed_value = parser.getfloat(section, option)
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be a float.')
            elif option_type == bool:
                try:
                    typed_value = parser.getboolean(section, option)
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be a boolean (True/False).')
            elif get_origin(option_type) == list:
                item_type, = get_args(option_type)
                lines = parser.get(section, option).split('\n')
                converter = self._parse_bool if item_type == bool else item_type
                try:
                    typed_value = [converter(line) for line in lines if line != '']
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be a list of {option_type}, one per line.')
            elif get_origin(option_type) == Literal:
                allowed_values = get_args(option_type)
                typed_value = parser.get(section, option)
                if typed_value not in allowed_values:
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {typed_value}. Must be one of: {", ".join(allowed_values)}.')
            else:
                raise NotImplementedError(f'Option type: {option_type}')
            self.__setattr__(option, typed_value)
    
    @staticmethod
    def _parse_bool(string: str) -> bool:
        low_string = string.lower()
        if low_string in ('true', '1'):
            return True
        elif low_string in ('false', '0'):
            return False
        else:
            raise ValueError(f'Not a boolean: {string}')
    
    @staticmethod
    def _is_string_literal(typ: Type) -> bool:
        return get_origin(typ) == Literal and all(isinstance(v, str) for v in get_args(typ))

class Config(object):
    '''Represents configuration like in .ini file.
    Subclasses of Config can declare instance variables corresponding to individual configuration sections, these should be subclasses of ConfigSection.
    Instance variables with prefix _ are ignored.
    '''

    __SECTION_TYPE: Final = ConfigSection
    __section_types: Dict[str, Type[ConfigSection]]

    def __init__(self, filename: Optional[str] = None, allow_extra: bool = False, allow_missing: bool = False):
        '''Create new Config object with either default values or loaded from an .ini file.'''
        cls = type(self)
        self.__section_types = {section: typ for section, typ in get_type_hints(cls).items() if not section.startswith(f'_')}
        for section, typ in self.__section_types.items():
            assert issubclass(typ, self.__SECTION_TYPE)
            value = vars(cls).get(section, typ())
            self.__setattr__(section, value)
        if filename is not None:
            self.load_from_file(filename, allow_extra=allow_extra, allow_missing=allow_missing)

    def __str__(self) -> str:
        lines = []
        for section in vars(self):
            if not section.startswith('_'):
                value = self.__getattribute__(section)
                lines.append(f'[{section}]')
                lines.append(str(value))
                lines.append('')
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        sects = []
        for section in vars(self):
            if not section.startswith('_'):
                value = self.__getattribute__(section)
                sects.append(f'{section}={repr(value)}')
        return f'{type(self).__name__}({", ".join(sects)})'
    

    def load_from_file(self, filename: str, allow_extra: bool = False, allow_missing: bool = False) -> None:
        '''Load configuration options from an .ini file into this Config object.'''
        parser = configparser.ConfigParser()
        with open(filename) as r:
            parser.read_file(r)
        loaded_sections = parser.sections()

        if not allow_extra:
            for section in loaded_sections:
                assert section in self.__section_types, f'Extra section [{section}] in {filename}'
        else:
            for section in loaded_sections:
                if section not in self.__section_types:
                    self.__section_types[section] = self.__SECTION_TYPE
                    self.__setattr__(section, self.__SECTION_TYPE())

        if not allow_missing:
            for section in self.__section_types:
                assert section in loaded_sections, f'Missing section [{section}] in {filename}'
            
        for section in loaded_sections:
            self.__getattribute__(section)._set_options(parser, section, allow_extra=allow_extra, allow_missing=allow_missing, filename=filename)


@dataclass
class FilePath(os.PathLike, object):
    '''Represents a path to a file or directory.
    full = dir/base = dir/name+ext
    '''
    full: str  # full path
    dir: str  # full directory path
    base: str  # basename
    name: str  # basename without extension
    ext: str  # extension

    def __init__(self, the_path: Union[str, 'FilePath'], *subpaths: Union[str, 'FilePath']):
        self.full = path.join(str(the_path), *map(str, subpaths))
        self.dir, self.base = path.split(self.full)
        self.name, self.ext = path.splitext(self.base)

    def __repr__(self) -> str:
        full_path = self.full.rstrip('/')
        d = '/' if self.isdir() else ''
        return f'FilePath({full_path}{d})'
    
    def __str__(self) -> str:
        full_path = self.full.rstrip('/')
        d = '/' if self.isdir() else ''
        return f'{full_path}{d}'
    
    def __fspath__(self) -> str:
        return str(self)

    @staticmethod
    def string(file_path: Union['FilePath', str, None]) -> Optional[str]:
        return str(file_path) if file_path is not None else None

    def abs(self) -> 'FilePath':
        '''Get equivalent absolute path'''
        pwd = os.getcwd()
        return FilePath(pwd, self)

    def parent(self) -> 'FilePath':
        '''Get path to the parent directory'''
        return FilePath(self.dir)

    def sub(self, *paths: str) -> 'FilePath':
        '''Create path relative to self (self/*paths)'''
        return FilePath(self.full, *paths)

    def isdir(self) -> bool:
        '''Is a directory?'''
        return path.isdir(self.full)

    def isfile(self) -> bool:
        '''Is a file?'''
        return path.isfile(self.full)

    def exists(self) -> bool:
        '''Exists?'''
        return path.exists(self.full)

    def ls(self, recursive: bool = False, only_files: bool = False, only_dirs: bool = False) -> List['FilePath']:
        '''List files in this directory or [] if self is not a directory.'''
        if recursive:
            result = list(self._ls_recursive())
        else:
            result = []
            if self.isdir():
                for file in os.listdir(self.full):
                    result.append(FilePath(self.full, file))
        if only_files:
            result = [f for f in result if f.isfile()]
        if only_dirs:
            result = [f for f in result if f.isdir()]
        return result
    
    def _ls_recursive(self, include_self: bool = False) -> Iterator['FilePath']:
        if include_self:
            yield self
        if self.isdir():
            for file in self.ls():
                yield from file._ls_recursive(include_self=True)

    def mkdir(self, *paths: str, **makedirs_kwargs) -> 'FilePath':
        '''Make directory self/*paths'''
        result = self.sub(*paths)
        os.makedirs(result.full, **makedirs_kwargs)
        return result
    
    def mv(self, dest: 'FilePath') -> 'FilePath':
        '''Move a file or directory ($ mv self dest)'''
        new_path = shutil.move(self.full, dest.full)
        return FilePath(new_path)

    def cp(self, dest: 'FilePath') -> 'FilePath':
        if self.isdir():
            raise NotImplementedError('Copying directories not implemented yet')
        else:
            new_path = shutil.copy(self.full, dest.full)
            return FilePath(new_path)

    def rm(self, recursive: bool = False, ignore_errors: bool = False) -> None:
        '''Remove this file or empty directory ($ rm self || rmdir self).
        If recursive, remove also non-empty directories ($ rm -r self).
        '''
        if ignore_errors:
            with suppress(OSError):
                return self.rm(recursive=recursive, ignore_errors=False)
        if self.isdir():
            if recursive:
                shutil.rmtree(self.full)
            else:
                os.rmdir(self.full)
        else:
            os.remove(self.full)

    def open(self, *args, **kwargs) -> TextIO:  #TODO replace by .write, .append, .dump_json, .load_json where possible
        '''Open file and return its file handle (like open(self)).'''
        return open(self.full, *args, **kwargs)
    
    def clear(self) -> 'FilePath':
        '''Remove all text from a file (or create empty file if does not exist).'''
        with self.open('w'):
            pass
        return self

    def glob(self, **kwargs) -> List['FilePath']:
        matches = glob.glob(self.full, **kwargs)
        return [FilePath(match) for match in matches]

    def archive_to(self, dest: 'FilePath') -> 'FilePath':
        fmt = dest.ext.lstrip('.')
        archive = shutil.make_archive(str(dest.parent().sub(dest.name)), fmt, str(self))
        return FilePath(archive)



class RedirectIO:
    def __init__(self, stdin: Union[FilePath, str, None] = None, stdout: Union[FilePath, str, None] = None, stderr: Union[FilePath, str, None] = None, 
                 tee_stdout: Union[FilePath, str, None] = None, tee_stderr: Union[FilePath, str, None] = None, 
                 append_stdout: bool = False, append_stderr: bool = False):
        assert stdout is None or tee_stdout is None, f'Cannot specify both stdout and tee_stdout'
        assert stderr is None or tee_stderr is None, f'Cannot specify both stderr and tee_stderr'
        self.new_in_file = FilePath.string(stdin)
        self.new_out_file = FilePath.string(stdout)
        self.new_err_file = FilePath.string(stderr)
        self.tee_out_file = FilePath.string(tee_stdout)
        self.tee_err_file = FilePath.string(tee_stderr)
        self.append_stdout = append_stdout
        self.append_stderr = append_stderr

    def __enter__(self):
        out_mode = 'a' if self.append_stdout else 'w'
        err_mode = 'a' if self.append_stderr else 'w'
        if self.new_in_file is not None:
            self.new_in = open(self.new_in_file, 'r')
            self.old_in = sys.stdin
            sys.stdin = self.new_in
        if self.new_out_file is not None:
            self.new_out = open(self.new_out_file, out_mode)
            self.old_out = sys.stdout
            sys.stdout = self.new_out
        if self.new_err_file is not None:
            self.new_err = open(self.new_err_file, err_mode)
            self.old_err = sys.stderr
            sys.stderr = self.new_err
        if self.tee_out_file is not None:
            self.new_out = Tee(sys.stdout, open(self.tee_out_file, out_mode))  # TODO close stream!
            self.old_out = sys.stdout
            sys.stdout = self.new_out
        if self.tee_err_file is not None:
            self.new_err = Tee(sys.stderr, open(self.tee_err_file, err_mode))  # TODO close stream!
            self.old_err = sys.stderr
            sys.stderr = self.new_err

    def __exit__(self, exctype, excinst, exctb):
        if self.new_in_file is not None:
            sys.stdin = self.old_in
            self.new_in.close()
        if self.new_out_file is not None:
            sys.stdout = self.old_out
            self.new_out.close()
        if self.new_err_file is not None:
            sys.stderr = self.old_err
            self.new_err.close()
        if self.tee_out_file is not None:
            sys.stdout = self.old_out
            self.new_out.outputs[1].close()
        if self.tee_err_file is not None:
            sys.stderr = self.old_err
            self.new_err.outputs[1].close()


class Job(NamedTuple):
    name: str
    func: Callable
    args: Sequence
    kwargs: Mapping
    stdout: FilePath
    stderr: FilePath

class JobResult(NamedTuple):
    job: Job
    result: Any
    worker: str

def run_jobs_with_multiprocessing(jobs: Sequence[Job], n_processes: Optional[int] = None, progress_bar: bool = False, callback: Optional[Callable[[JobResult], Any]] = None, pool: Optional[multiprocessing.Pool] = None) -> List[JobResult]:
    '''Run jobs (i.e. call job.func(*job.args, **job.kwargs)) in n_processes processes. 
    Standard output and standard error output are saved in files job.stdout and job.stderr.
    Default n_processes: number of CPUs.
    If n_processes==1, then run jobs sequentially without starting new processes (useful for debugging).'''
    if n_processes is None and pool is None:
        n_processes = multiprocessing.cpu_count()
    n_jobs = len(jobs)
    results = []
    with ProgressBar(n_jobs, title=f'Running {n_jobs} jobs in {n_processes} processes', mute = not progress_bar) as bar:
        if pool is not None:
            result_iterator = pool.imap_unordered(_run_job, jobs)
            for result in result_iterator:
                if callback is not None:
                    callback(result)
                results.append(result)
                bar.step()
        elif n_processes == 1:
            for job in jobs:
                result = _run_job(job)
                if callback is not None:
                    callback(result)
                results.append(result)
                bar.step()
        else:
            with multiprocessing.Pool(n_processes) as ad_hoc_pool:
                result_iterator = ad_hoc_pool.imap_unordered(_run_job, jobs)
                for result in result_iterator:
                    if callback is not None:
                        callback(result)
                    results.append(result)
                    bar.step()
    return results
    
def _run_job(job: Job) -> JobResult:
    worker = multiprocessing.current_process().name
    with RedirectIO(stdout=job.stdout, stderr=job.stderr):
        result = job.func(*job.args, **job.kwargs)
    return JobResult(job=job, result=result, worker=worker)



def test_NumpyMinHeap():
    ks = np.random.permutation(range(100))
    print(ks)
    heap = NumpyMinHeap((), int, 100, keys_values=zip(ks,-ks))
    print(heap.keys)
    print(heap.values)
    while not heap.is_empty():
        print(heap.pop_min())