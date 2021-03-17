import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Union


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

class Structure(dict):
    @property
    def symbol(self): 
        '''Array of atom element symbols (numpy array)'''
        return self['symbol']
    @property
    def name(self): 
        '''Array of atom names (numpy array)'''
        return self['name']
    @property
    def resn(self):
        '''Array of residue names (numpy array)'''
        return self['resn']
    @property
    def resi(self): 
        '''Array of residue indices (numpy array)'''
        return self['resi']
    @property
    def chain(self): 
        '''Array of chain identifiers (numpy array)'''
        return self['chain']
    @property
    def auth_chain(self): 
        '''Array of auth chain identifiers (numpy array)'''
        return self['auth_chain']
    @property
    def entity(self): 
        '''Array of entity identifiers (numpy array)'''
        return self['entity']
    @property
    def coords(self): 
        '''XYZ-coordinates (numpy array with shape (3, N), one atom = one column)'''
        return self['coords']
    @coords.setter
    def coords(self, value):
        self['coords'] = value
    @property
    def count(self): 
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
        self['resi'] = np.array(resi)
        self['chain'] = np.array(chain)
        self['auth_chain'] = np.array(auth_chain)
        self['entity'] = np.array(entity)
        self['coords'] = np.array(coords)
        if coords.shape[0] != 3:
            raise ValueError(f'coords must have shape (3, N), passed value has shape {coords.shape}')
        # print(id)
        lengths = [values.shape[-1] for values in self.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f'All arguments must have the same length (encountered lengths: {lengths})')
    
    def add_field(self, field_name: str, values: Union[List[Any], np.ndarray]) -> None:
        assert len(values) == self.count
        self[field_name] = np.array(values)

    def filter(self, mask: np.ndarray) -> 'Structure':
        '''Select atoms for which mask[i] is True and return them as a new Structure.'''
        new_struct_dict = {}
        for field, values in self.items():
            new_struct_dict[field] = values[..., mask]
        new_struct = Structure(**new_struct_dict)
        return new_struct

    def filter_by_field(self, field_name: str, value: object) -> 'Structure':
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

    def get_alpha_trace(self) -> 'Structure':
        mask = np.logical_and.reduce((self.symbol == 'C', self.name == 'CA', self.resi != None))  # type: ignore
        return self.filter(mask)

    def get_sequence(self, assume_alpha_trace=False) -> str:
        if assume_alpha_trace:
            struct = self
        else:
            struct = self.get_alpha_trace()
        # sequence = ' '.join(struct.resn)
        sequence = ''.join(_LONG2SHORT[l] for l in struct.resn)
        return sequence

    def to_cif(self) -> str:
        lines = []
        columns = []
        name = 'structure'
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

