'''
General-purpose functions.
'''

from __future__ import annotations
from pathlib import Path
import sys
import json
import re
import heapq
import numpy as np
import hashlib
from typing import List, Tuple, Dict, Iterable, TypeVar, Optional, Generic, Callable, TextIO


COMMENT_SYMBOL = '#'
RE_DTYPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*dtype\s*=\s*(\w+)\s*$')
RE_SHAPE = re.compile('^\s*'+re.escape(COMMENT_SYMBOL)+'\s*shape\s*=\s*(\w+)\s*,\s*(\w+)\s*$')

K = TypeVar('K')
V = TypeVar('V')


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()  # necessary when importing pymol because it somehow fucks up stdout

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

def single(iterable: Iterable[V]) -> V:
    '''Return the single element of `iterable`, or raise ValueError if len(iterable) != 1'''
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
    """Do the same as sep.join(elements), but automatically convert elements to strings and include at most `three_dots_after` elements. 
    Put '...' at the end if there are more elements"""
    if three_dots_after == -1 or len(elements) > three_dots_after:
        return sep.join(str(elem) for  elem in elements[:three_dots_after]) + '...'
    else:
        return sep.join(str(elem) for elem in elements)

def insert_after(dictionary: Dict[K, V], after_what: K, new_key_value_pairs: Iterable[Tuple[K, V]]) -> None:
    '''Insert a new key-value pair into the dictionary just after key `after_what`.
    Involves clearing and refilling the dictionary!
    '''
    key_value_pairs = list(dictionary.items())
    dictionary.clear()
    for key, value in key_value_pairs:
        dictionary[key] = value
        if key == after_what:
            for k, v in new_key_value_pairs:
                dictionary[k] = v

def first_index_where(items: list[V], predicate: Callable[[V], bool]) -> int:
    '''Return index of the first item fulfulling the predicate, -1 if there is no such item.'''
    for i, item in enumerate(items):
        if predicate(item):
            return i
    return -1

def read_matrix(filename: Path, sep='\t', dtype=None) -> Tuple[np.ndarray, List[str], List[str]]:
    with open(filename) as f:
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

def integer_suffix(string: str, empty_means_zero: bool = True) -> int:
    '''Return the longest numeric suffix of `string`.'''
    for i in range(len(string)):
        try:
            return int(string[i:])
        except ValueError:
            pass
    if empty_means_zero:
        return 0
    else:
        raise ValueError(f'String {string} contains no integer suffix')

def consistent_hash(string: str) -> str:
    '''Returns a hash of a string. The hash itself is a string with 40 hexadecimal characters.'''
    return hashlib.sha1(string.encode('utf8')).hexdigest()

def consistent_pseudorandom_choice(strings: Iterable[str], size: int) -> List[int]:
    '''Return indices of pseudorandomly selected strings. The number of selected indices is min(size, len(strings)).
    The selection is done in such way that:
    - For the same input set of strings (even permutated) and same size, the resulting string set is always the same (and in the same order).
    - If i <= j, then consistent_pseudorandom_choice(S, i) is a subset of consistent_pseudorandom_choice(S, j).
    - Similar string sets give similar resulting string set.
    '''
    hashis = [(consistent_hash(string), i) for i, string in enumerate(strings)]
    hashis.sort()
    return [i for (hashsh, i) in hashis[:size]]

def int_or_all(string: str) -> int|None:
    '''Parse string as integer or return None if it is 'all'.
    >>> int_or_all('123')
    123
    >>> int_or_all('all')
    None
    '''
    if string.strip() == 'all':
        return None
    else: 
        try:
            return int(string)
        except ValueError:
            raise ValueError(f"invalid value: {repr(string)}, must be an integer or 'all'")


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


def dump_json(obj: object, file: TextIO|Path, minify: bool = False) -> None:
    is_writer = hasattr(file, 'write')
    options: Dict[str, object]
    if minify:
        options = {'separators': (',', ':')}
    else:
        options = {'indent': 2}
    if is_writer:
        json.dump(obj, file, **options)  # type: ignore
        file.write('\n')  # type: ignore
    else:
        assert isinstance(file, Path)
        with open(file, 'w') as w:  # type: ignore
            json.dump(obj, w, **options)  # type: ignore
            w.write('\n')
