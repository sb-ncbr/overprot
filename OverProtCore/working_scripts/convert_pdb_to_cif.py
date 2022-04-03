'''
This Python script converts structures in PDB format to mmCIF format.

Example usage:
    python3  -m convert_pdb_to_cif  example.pdb  example.cif
    python3  -m convert_pdb_to_cif  examples_pdb/ examples_cif/
'''

from __future__ import annotations
import sys
from pathlib import Path

from overprot.libs import lib_pymol
from overprot.libs.lib_cli import cli_command, run_cli_command


#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

@cli_command()
def main(input_file: Path, output_file: Path) -> int|None:
    # TODO add parameters
    '''Convert structures in PDB format to mmCIF format.
    @param  input_file   Input PDB file or directory with PDB files
    @param  output_file  Output CIF file or directory for CIF files
    '''
    converted = 0
    if input_file.is_file():
        if input_file.suffix != '.pdb' or output_file.suffix != '.cif':
            print(f'Input file ({input_file}) must be a .pdb file and output file must be a .cif file (or both can be directories).', file=sys.stderr)
            return 1
        lib_pymol.convert_file(input_file, output_file)
        converted += 1
    elif input_file.is_dir():
        output_file.mkdir(exist_ok=True, parents=True)
        for ifile in input_file.glob('*.pdb'):
            if ifile.is_file():
                ofile = output_file / f'{ifile.stem}.cif'
                lib_pymol.convert_file(ifile, ofile)
                converted += 1
    else:
        print(f'Input file ({input_file}) must be a .pdb file or a directory.', file=sys.stderr)
        return 1
    s = 's' if converted != 1 else ''
    print(f'Converted {converted} file{s}', file=sys.stderr)
    return 0

if __name__ == '__main__':
    run_cli_command(main)
