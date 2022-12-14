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


def make_summaries(pdbs: list[str], chain_outdir: Path|None, residue_outdir: Path|None, structure_sources: list[str], 
        residue_summaries_skip_protein: bool = False, failures_file: Path|None = None) -> None:
    if chain_outdir is not None:
        chain_outdir.mkdir(parents=True, exist_ok=True)
    if residue_outdir is not None:
        residue_outdir.mkdir(parents=True, exist_ok=True)
    if len(pdbs) == 0:
        return
    if chain_outdir is None and residue_outdir is None:
        return
    js = [{"domain": ".", "pdb": pdb, "chain_id": None, "ranges": None} for pdb in pdbs]
    sample_file = (chain_outdir or residue_outdir)/f'sample.json'
    with open(sample_file, 'w') as w:
        json.dump(js, w)
    args = [STRUCTURE_CUTTER_DLL, sample_file]
    if chain_outdir is not None:
        args.append('--summary_outdir')
        args.append(chain_outdir)
    if residue_outdir is not None:
        args.append('--residue_summary_outdir')
        args.append(residue_outdir)
        if residue_summaries_skip_protein:
            args.append('--residue_summary_skip_protein')
    if len(structure_sources) > 0:
        args.append('--sources')
        args.append(' '.join(structure_sources))
    if failures_file is not None:
        args.append('--failures')
        args.append(failures_file)
    with RedirectIO(stdout=os.devnull):
        lib_run.run_dotnet(*args, timing=True)
    sample_file.unlink()

def append_file(source: Path, dest: Path, remove_source: bool = False):
    with open(source, 'rb') as r:
        with open(dest, 'ab') as w:
            w.write(r.read())
    if remove_source:
        source.unlink()

@cli_command(parsers={'structure_sources': str.split})
def main(pdb_list: Path, chain_outdir: Path, residue_outdir: Path, structure_sources: list[str], 
        residue_summaries_skip_protein: bool = False, breakdown: bool = False, save_failures: bool = False, processes: Optional[int] = None) -> Optional[int]:
    '''Create chain summaries for a list of PDB entries.
    @param  `pdb_list`  File with PDB entry list (PDB IDs separated by whitespace).
    @param  `chain_outdir`       Directory for output files (chain summaries).
    @param  `residue_outdir`     Directory for output files (residue summaries).
    @param  `structure_sources`  List of structure sources, e.g. 'http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif.gz'.
                                 (From command line, separate the sources by space within the argument.)
    @param  `breakdown`      Split the output files into subdirectories based on the middle 2 characters of the PDB ID (e.g. 1tqn -> outdir/tq/1tqn.json).
    @param  `save_failures`  Save the list of failed PDBs in outdir/failures.txt (default: )
    @param  `processes`      Number of processes to run. [default: number of CPUs]
    '''
    text = pdb_list.read_text()
    pdbs = text.split()
    for pdb in pdbs:
        if not pdb.isalnum():
            raise ValueError(f"PDB ID must be alphanumeric, not '{pdb}'")
    chain_outdir.mkdir(parents=True, exist_ok=True)
    residue_outdir.mkdir(parents=True, exist_ok=True)
    if breakdown:
        subsets = defaultdict(list)
        for pdb in pdbs:
            sub = pdb[1:3]
            subsets[sub].append(pdb)
        jobs = []
        for subdir, subset in subsets.items():
            kwargs = {'failures_file': chain_outdir/subdir/'failures.txt'} if save_failures else {}
            jobs.append(lib_multiprocessing.Job(
                name=subdir, 
                func=make_summaries, 
                args=(subset, chain_outdir/subdir, residue_outdir/subdir, structure_sources),
                kwargs={'residue_summaries_skip_protein': residue_summaries_skip_protein}
            ))
        print(f'Making chain summaries for {len(pdbs)} PDBs:')
        def append_failures(subdir: str) -> None:
            if save_failures:
                append_file(chain_outdir/subdir/'failures.txt', chain_outdir/'failures.txt', remove_source=True)
        lib_multiprocessing.run_jobs_with_multiprocessing(jobs, n_processes=processes, progress_bar=True, callback = append_failures)
    else:
        make_summaries(pdbs, chain_outdir, residue_outdir, structure_sources, residue_summaries_skip_protein=residue_summaries_skip_protein)
    return 0


if __name__ == '__main__':
    run_cli_command(main)
