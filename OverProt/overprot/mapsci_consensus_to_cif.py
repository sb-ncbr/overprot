'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import numpy as np
import argparse
from typing import Dict, Any, Optional, Union

from .libs import superimpose3d
from .libs.lib import FilePath

#  CONSTANTS  ################################################################################

CATEGORY = '_atom_site.'
DEFAULT_CHAIN = 'A'
DEFAULT_SYMBOL = 'C'
DEFAULT_ENTITY = 1

#  FUNCTIONS  ################################################################################

class AtomTable:
    def __init__(self):
        self.hetatm = []
        self.index = []
        self.name = []
        self.resn = []
        self.chain = []
        self.resi = []
        self.altloc = []
        self.x = []
        self.y = []
        self.z = []

    def count(self):
        return len(self.index)
        
    def range(self):
        return range(len(self.index))


def read_pdb_line(line):
    if line[0:6]=='ATOM  ':
        hetatm = False
    elif line[0:6]=='HETATM':
        hetatm = True
    else:
        return None
    index = line[6:11]
    name  = line[12:16]
    resn  = line[17:20]
    chain = line[21]
    resi  = line[22:26]
    altloc = line[26]
    x = line[30:38]
    y = line[38:46]
    z = line[46:54]
    return (hetatm, int(index), name, resn, chain, int(resi), altloc, float(x), float(y), float(z))

def read_pdb(filename: FilePath):
    table = AtomTable()
    with filename.open() as f:
        for line in iter(f.readline, ''):
            fields = read_pdb_line(line.strip('\n'))
            if fields != None:
                hetatm, index, name, resn, chain, resi, altloc, x, y, z = fields
                table.hetatm.append(hetatm)
                table.index.append(index)
                table.name.append(name)
                table.resn.append(resn)
                table.chain.append(chain)
                table.resi.append(resi)
                table.altloc.append(altloc)
                table.x.append(x)
                table.y.append(y)
                table.z.append(z)
    return table # AtomTable

def group_pdb_text(is_hetatm):
    return 'HETATM' if is_hetatm else 'ATOM'

def print_cif_minimal(atom_table, filename: Union[FilePath, str], structure_name='structure'):
    table = atom_table
    fields_values = [ ('group_PDB', ['HETATM' if het else 'ATOM' for het in table.hetatm]), 
                    ('id', table.index), ('type_symbol', table.symbol), ('label_atom_id', table.name), ('label_alt_id', table.altloc), 
                    ('label_comp_id', table.resn), ('label_asym_id', table.chain), ('label_entity_id', table.entity), ('label_seq_id', table.resi), ('auth_asym_id', table.chain), 
                    ('Cartn_x', table.x), ('Cartn_y', table.y), ('Cartn_z', table.z) ]
    with open(filename, 'w') as w:
        w.write('data_' + structure_name + '\n')
        w.write('loop_\n')
        for field, values in fields_values:
            w.write(CATEGORY + field + '\n')
        for i in table.range():
            for field, values in fields_values:
                value = f'{values[i]:.3f}' if field.startswith('Cartn_') else str(values[i])
                w.write(value if value.strip() != '' else '.')
                w.write(' ')
            w.write('\n')

def apply_laying_rotation_translation(atoms: AtomTable) -> None:
    coords = np.array([atoms.x, atoms.y, atoms.z])
    R, t = superimpose3d.laying_rotation_translation(coords, return_laid_coords=True)
    coords = superimpose3d.rotate_and_translate(coords, R, t)
    atoms.x = coords[0]
    atoms.y = coords[1]
    atoms.z = coords[2]


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_pdb', help='Consensus PDB from MAPSCI', type=str)
    parser.add_argument('output_cif', help='File for CIF output', type=str)
    args = parser.parse_args()
    return vars(args)


def main(input_pdb: Union[FilePath, str], output_cif: Union[FilePath, str]) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    input_pdb = FilePath(input_pdb)
    atoms = read_pdb(input_pdb)
    atoms.entity = [DEFAULT_ENTITY] * atoms.count()
    atoms.chain = [DEFAULT_CHAIN] * atoms.count()
    atoms.symbol = [DEFAULT_SYMBOL] * atoms.count()
    apply_laying_rotation_translation(atoms)  # Place starting and ending more in front, place starting more top-left
    print_cif_minimal(atoms, output_cif, structure_name='consensus')
    return None


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)
