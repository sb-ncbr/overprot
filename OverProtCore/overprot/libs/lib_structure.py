'''
Representation of protein structures (sets of atoms in 3D)
'''

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Any, Literal, Sequence
from numpy.typing import NDArray, ArrayLike


_SHORT2LONG = {'X': 'XXX', 'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 
               'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 
               'Q': 'GLN', 'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
_LONG2SHORT = defaultdict(lambda: 'X', {l: s for s, l in _SHORT2LONG.items()})

_CIF_CATEGORY_NAME = 'atom_site'
_CIF_FIELD_NAMES = {'group': 'group_PDB',
                    'id': 'id',
                    'symbol': 'type_symbol',
                    'name': 'label_atom_id',
                    'resn': 'label_comp_id',
                    'resi': 'label_seq_id',
                    'chain': 'label_asym_id',
                    'auth_chain': 'auth_asym_id',
                    'entity': 'label_entity_id',
                    'alt': 'label_alt_id',
                    'coords': ('Cartn_x', 'Cartn_y', 'Cartn_z')
                    }
_CIF_FIELD_FORMATS = {'coords': '{:.3f}'}

_CUT_THRESHOLD = 8
_SMOOTHING_PROFILE = (6980/2**16, 16384/2**16, 18808/2**16, 16384/2**16, 6980/2**16)
# # Calculating smart profile of length 5 (should nicely smooth 3.6-helix and 2.0-helix):
# A = np.array([[1, 2, 2], [1, -2, 2], [1, 2*np.cos(np.pi/1.8), 2*np.cos(2*np.pi/1.8)]])
# b = [1, 0, 0]
# x = np.linalg.solve(A, b)
# _SMOOTHING_PROFILE = [x[2], x[1], x[0], x[1], x[2]]
# + binary round to sum to 1 exactly

COORDS_TYPE = np.float32
RESI_TYPE = np.int32
RESI_NULL: int = np.iinfo(RESI_TYPE).min


class Structure(dict):
    '''Represent a 3D structure of a protein'''
    @property
    def symbol(self) -> NDArray: 
        '''Array of atom element symbols (numpy array)'''
        return self['symbol']
    @symbol.setter
    def symbol(self, value: ArrayLike):
        self['symbol'] = _as_array(value)    

    @property
    def name(self) -> NDArray: 
        '''Array of atom names (numpy array)'''
        return self['name']
    
    @property
    def resn(self) -> NDArray:
        '''Array of residue names (numpy array)'''
        return self['resn']
    
    @property
    def resi(self) -> NDArray[RESI_TYPE]: 
        '''Array of residue indices (numpy array)'''
        return self['resi']
    @resi.setter
    def resi(self, value: ArrayLike):
        self['resi'] = _as_array(value, RESI_TYPE, none_replacement=RESI_NULL)
    
    @property
    def chain(self) -> NDArray: 
        '''Array of chain identifiers (numpy array)'''
        return self['chain']
    @chain.setter
    def chain(self, value: ArrayLike):
        self['chain'] = _as_array(value)    

    @property
    def auth_chain(self) -> NDArray: 
        '''Array of auth chain identifiers (numpy array)'''
        return self['auth_chain']
    @auth_chain.setter
    def auth_chain(self, value: ArrayLike):
        self['auth_chain'] = _as_array(value)

    @property
    def entity(self) -> NDArray: 
        '''Array of entity identifiers (numpy array)'''
        return self['entity']
    
    @property
    def coords(self) -> NDArray[COORDS_TYPE]: 
        '''XYZ-coordinates (numpy array with shape (3, N), one atom = one column)'''
        return self['coords']
    @coords.setter
    def coords(self, value: ArrayLike):
        self['coords'] = _as_array(value, dtype=COORDS_TYPE)
    
    @property
    def count(self) -> int: 
        '''Number of atoms'''
        return self.symbol.shape[0]
    
    def __init__(self, *, symbol, name, resn, resi, chain, auth_chain, entity, coords, id=None, alt=None, group=None):
        n = len(symbol)
        self['group'] = np.array(group) if group is not None else np.full(n, 'ATOM')
        self['id'] = np.array(id) if id is not None else np.arange(1, n+1)
        self['symbol'] = np.array(symbol)
        self['name'] = np.array(name)
        self['alt'] = np.array(alt) if alt is not None else np.full(n, '.')
        self['resn'] = np.array(resn)
        self.resi = np.array(resi)
        self['chain'] = np.array(chain)
        self['auth_chain'] = np.array(auth_chain)
        self['entity'] = np.array(entity)
        self.coords = coords
        if coords.shape[0] != 3:
            raise ValueError(f'coords must have shape (3, N), passed value has shape {coords.shape}')
        lengths = [values.shape[-1] for values in self.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f'All arguments must have the same length (encountered lengths: {lengths})')
    
    def add_field(self, field_name: str, values: list[Any]|np.ndarray) -> None:
        assert len(values) == self.count
        self[field_name] = np.array(values)

    def filter(self, mask: np.ndarray) -> Structure:
        '''Select atoms for which mask[i] is True and return them as a new Structure.'''
        new_struct_dict = {}
        for field, values in self.items():
            new_struct_dict[field] = values[..., mask]
        new_struct = Structure(**new_struct_dict)
        return new_struct

    def filter_by_field(self, field_name: str, value: object) -> Structure:
        '''Select atoms for which the value of field_name is equal to value and return them as a new Structure.'''
        mask = self[field_name] == value
        return self.filter(mask)

    def is_alpha_trace(self) -> bool:
        '''Decide whether the structure is an alpha-trace, i.e. contains only non-hetatm C-alpha atoms.'''
        if not all(self.symbol == 'C'):
            return False
        if not all(self.name == 'CA'):
            return False
        if not all(self.resi != None):
            return False
        return True

    def get_alpha_trace(self, remove_repeating_resi=False) -> Structure:
        mask = np.logical_and.reduce((self.symbol == 'C', self.name == 'CA', self.resi != RESI_NULL))  # type: ignore
        result = self.filter(mask)
        if remove_repeating_resi:
            nonrep_mask = np.ones(result.count, dtype=np.bool_)
            nonrep_mask[1:] = result.resi[1:] != result.resi[:-1]
            result = result.filter(nonrep_mask)
        return result

    def smooth_trace(self, profile=_SMOOTHING_PROFILE, cut_threshold=_CUT_THRESHOLD) -> Structure:
        '''Smooth the positions of atoms (asserts `self` is an alpha trace).'''
        result: Structure = self.copy()
        result.coords = _smooth(self.coords, profile=profile, ends='straight', cut_threshold=cut_threshold)
        return result

    def get_only_polymer(self) -> Structure:
        return self.filter(self.resi != RESI_NULL)

    def get_sequence(self, assume_alpha_trace=False) -> str:
        if assume_alpha_trace:
            struct = self
        else:
            struct = self.get_alpha_trace()
        sequence = ''.join(_LONG2SHORT[l] for l in struct.resn)
        return sequence

    def copy(self) -> Structure:
        return Structure(**self) 

    def to_cif(self, name='structure') -> str:
        lines = []
        columns = []
        lines.append(f'data_{name}')
        lines.append('loop_')
        for field, values in self.items():
            cif_field = _CIF_FIELD_NAMES.get(field, field)
            if len(values.shape) == 1:
                lines.append(f'_{_CIF_CATEGORY_NAME}.{cif_field}')
                columns.append(values)
            else:
                assert values.shape == (len(cif_field), self.count)
                for i in range(len(cif_field)):
                    lines.append(f'_{_CIF_CATEGORY_NAME}.{cif_field[i]}')
                    columns.append(values[i])
        for i in range(self.count):
            lines.append(' '.join(str(col[i]) for col in columns))
        return '\n'.join(lines)

    def save_cif(self, filename: Path, name='structure') -> None:
        filename.write_text(self.to_cif(name=name))

    @staticmethod
    def from_pdb(filename: Path) -> Structure:
        symbols = []
        names = []
        resns = []
        resis = []
        chains = []
        auth_chains = []
        entities = []
        xs = []
        ys = []
        zs = []
        ids = []
        alts = []
        groups = []
        chain2entity = {}
        with open(filename) as f:
            for line in iter(f.readline, ''):
                fields = _read_pdb_line(line)
                if fields != None:
                    group, index, name, resn, chain, resi, alt, x, y, z, symbol = fields
                    symbols.append(symbol)
                    names.append(name)
                    resns.append(resn)
                    resis.append(resi)
                    chains.append(chain)
                    auth_chains.append(chain)
                    if chain not in chain2entity:
                        chain2entity[chain] = len(chain2entity) + 1
                    entities.append(chain2entity[chain])
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    ids.append(index)
                    alts.append(alt)
                    groups.append(group)
        return Structure(symbol=symbols, name=names, resn=resns, resi=resis, chain=chains, 
            auth_chain=auth_chains, entity=entities, coords=np.array([xs, ys, zs]), 
            id=ids, alt=alts, group=groups)


def _as_array(value: ArrayLike, dtype=None, none_replacement=None) -> NDArray:
    if none_replacement is None:
        return np.asarray(value, dtype=dtype)
    else:
        value = np.asarray(value)
        value[value==None] = none_replacement
        return np.asarray(value, dtype=dtype)

def _smooth(coords: NDArray[COORDS_TYPE], *, profile: Sequence[float]=_SMOOTHING_PROFILE, ends: Literal['original', 'mirror', 'straight'] = 'straight', cut_threshold: float = np.inf, out: NDArray[COORDS_TYPE] = None) -> NDArray[COORDS_TYPE]:
    '''Cut a sequence of N points `coords` (shape (3, N)) into chunks (cut where point distance is above cut_threshold).
    Then apply smoothing to each chunk. Meaning of `ends` - see _smooth_chunk.
    '''
    n_points = coords.shape[1]
    offsets: NDArray|Sequence[int]
    if cut_threshold == np.inf:
        offsets = (0, n_points)
    else:
        offsets = _cut_chain(coords, cut_threshold)
    n_chunks = len(offsets) - 1
    k = len(profile)
    if out is None:
        out = np.empty_like(coords)
    for i in range(n_chunks):
        fro, to = offsets[i], offsets[i+1]
        _smooth_chunk(coords[:, fro:to], profile=profile, ends=ends, out=out[:, fro:to])
    return out
     
def _smooth_chunk(coords: NDArray[COORDS_TYPE], *, profile: Sequence[float], ends: Literal['trim', 'original', 'mirror', 'straight'], out: NDArray[COORDS_TYPE]) -> None:
    '''Smooth a sequence of N points `coords` (shape (3, N)) by convolution with `profile` (shape (k,)). Save the result in `out`.
    `ends` controls the dealing with first and last points, which don't have enough neighbors to calculate convolution
    ('trim' shortens the sequence to (3, N-k+1); 'original' keeps unsmoothed coordinates, 'straight' interpolates to keep the shape (3, N)).
    If the sequence is too short to smooth, keep unsmoothed coordinates.'''
    # scipy.signal.oaconvolve() might be faster.
    n_points = coords.shape[1]
    k = len(profile)
    assert k % 2 == 1, 'The length of `profile` must be odd'
    kk = k // 2
    if ends == 'trim':
        assert n_points >= k
        out[:, :] = 0
        for i in range(k):
            out += profile[i] * coords[:, i: -2*kk+i or None]
        out /= sum(profile)
    else:
        min_length = k if ends=='original' else k+kk if ends=='mirror' else k+1
        if n_points < min_length:
            out[:, :] = coords
            return
        _smooth_chunk(coords, profile=profile, ends='trim', out=out[:, kk:-kk])
        if ends == 'original':
            out[:, :kk] = coords[:, :kk]
            out[:, -kk:] = coords[:, -kk:]
        elif ends == 'mirror':
            out[:, :kk] = 2*out[:, kk:kk+1] - out[:, 2*kk:kk:-1]
            out[:, -kk:] = 2*out[:, -kk-1:-kk] - out[:, -kk-2:-2*kk-2:-1]
        elif ends == 'straight':
            start = out[:, kk]
            startstep = start - out[:, kk+1]
            end = out[:, -kk-1]
            endstep = end - out[:, -kk-2]
            out[:, :kk] = start.reshape((-1, 1)) + startstep.reshape((-1, 1)) * np.arange(kk, 0, -1).reshape((1, -1))
            out[:, -kk:] = end.reshape((-1, 1)) + endstep.reshape((-1, 1)) * np.arange(1, kk+1).reshape((1, -1))
        else: 
            raise NotImplementedError()
   
def _running_distance(coords: NDArray[COORDS_TYPE]) -> NDArray[COORDS_TYPE]:
    '''Return distances between each two neighbouring points.'''
    diff = coords[:,1:] - coords[:,:-1]
    return np.sqrt(np.sum(diff**2, axis=0))

def _cut_chain(coords: NDArray[COORDS_TYPE], distance_threshold: float = 8) -> NDArray[np.int_]:
    '''Return indices where the chain should be cut, e.g. [0 5 8 12] means "cut to [0 1 2 3 4], [5 6 7], [8 9 10 11]".'''
    dist = _running_distance(coords)
    cut_indices = np.where(dist > distance_threshold)[0]
    offsets = np.hstack([[0], cut_indices+1, [coords.shape[1]]])
    return offsets

def _read_pdb_line(line: str) -> tuple|None:
    group = line[0:6].strip()
    if group not in ['ATOM', 'HETATM']:
        return None
    index = int(line[6:11])
    name  = line[12:16].strip() or '.'
    resn  = line[17:20].strip() or '.'
    chain = line[21].strip() or '.'
    resi  = int(line[22:26])
    altloc = line[26].strip() or '.'
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    symbol = line[76:78].strip() or '.'
    return (group, index, name, resn, chain, resi, altloc, x, y, z, symbol)
