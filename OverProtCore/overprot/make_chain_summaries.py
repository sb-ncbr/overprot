'''
Create chain summaries for a list of PDB entries.
All lines before a line starting with --- are treated as header.
PDB ID is selected as the first column (whitespace-separated) in the file.

Example usage:
    python3  -m overprot.make_chain_summaries  --help
'''

from __future__ import annotations
import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

from .libs import lib_run
from .libs.lib_dependencies import STRUCTURE_CUTTER_DLL
from .libs import lib_multiprocessing
from .libs.lib_io import RedirectIO
from .libs.lib_logging import ProgressBar
from .libs.lib_cli import cli_command, run_cli_command


def make_summaries(pdbs: list[str], outdir: Path, structure_sources: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if len(pdbs) == 0:
        return
    js = [{"domain": ".", "pdb": pdb, "chain_id": ".", "ranges": ":"} for pdb in pdbs]
    sample_file = outdir/f'sample.json'
    with open(sample_file, 'w') as w:
        json.dump(js, w)
    args = [STRUCTURE_CUTTER_DLL, sample_file, '--summary_outdir', outdir]
    if len(structure_sources) > 0:
        args.append('--sources')
        args.append(' '.join(structure_sources))
    with RedirectIO(stdout=os.devnull):
        lib_run.run_dotnet(*args, timing=True)
    sample_file.unlink()

@cli_command(parsers={'structure_sources': str.split})
def main(pdb_list: Path, outdir: Path, structure_sources: list[str], breakdown: bool = False, processes: Optional[int] = None) -> Optional[int]:
    '''Create chain summaries for a list of PDB entries.
    @param  `pdb_list`  File with PDB entry list (PDB IDs separated by whitespace).
    @param  `outdir`    Directory for output files.
    @param  `structure_sources`  List of structure sources, e.g. 'http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif.gz'.
                                 (From command line, separate the sources by space within the argument.)
    @param  `breakdown`  Split the output files into subdirectories based on the middle 2 characters of the PDB ID (e.g. 1tqn -> outdir/tq/1tqn.json).
    @param  `processes`  Number of processes to run. [default: number of CPUs]
    '''
    text = pdb_list.read_text()
    pdbs = text.split()
    outdir.mkdir(parents=True, exist_ok=True)
    if breakdown:
        subsets = defaultdict(list)
        for pdb in pdbs:
            sub = pdb[1:3]
            subsets[sub].append(pdb)
        jobs = []
        for subdir, subset in subsets.items():
            jobs.append(lib_multiprocessing.Job(name=subdir, func=make_summaries, args=(subset, outdir/subdir, structure_sources)))
        print(f'Making chain summaries for {len(pdbs)} PDBs:')
        lib_multiprocessing.run_jobs_with_multiprocessing(jobs, n_processes=processes, progress_bar=True)
    else:
        make_summaries(pdbs, outdir, structure_sources)
    return 0


if __name__ == '__main__':
    run_cli_command(main)