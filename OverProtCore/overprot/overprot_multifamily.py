'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
import contextlib
import shutil
import json
from datetime import datetime

from .libs import lib
from .libs import lib_sh
from .libs import lib_multiprocessing
from .libs.lib_io import RedirectIO
from .libs.lib_logging import Timing, ProgressBar
from . import get_cath_family_list
from . import get_cath_example_domains
from . import get_cath_family_names
from . import domains_from_pdbeapi
from . import overprot

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################

HLINE = '-' * 60

def process_family(family: str, sample_size: int|str|None, directory: Path, config: Optional[Path] = None, domains: Optional[Path] = None) -> Optional[str]:
    '''Try running OverProt on a family. Return the error traceback if fails, None if succeeds.'''
    print(HLINE, family, sep='\n')
    print(HLINE, family, sep='\n', file=sys.stderr)
    try:
        overprot.main(family, sample_size, directory, config=config, domains=domains)
        return None
    except Exception as ex:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        return error

def family_callback(result: lib_multiprocessing.JobResult, directory: Path):
    family = result.job.name
    error = result.result
    with contextlib.suppress(OSError):
        lib_sh.cp(directory/'current'/f'{family}-out.txt', directory/'families'/family/'stdout.txt')
        lib_sh.cp(directory/'current'/f'{family}-err.txt', directory/'families'/family/'stderr.txt')
        lib_sh.mv(directory/'current'/f'{family}-out.txt', directory/'stdout_stderr'/ f'{family}-out.txt')
        lib_sh.mv(directory/'current'/f'{family}-err.txt', directory/'stdout_stderr'/ f'{family}-err.txt')
    if error is not None:
        with open(directory/'failed_families.txt', 'a') as w:
            print(family, file=w)
        with open(directory/'errors.txt', 'a') as w:
            print(family, error, sep='\n', end='\n\n', file=w)
    else:
        with open(directory/'succeeded_families.txt', 'a') as w:
            print(family, file=w)

def collect_results(families: List[str], input_dir: Path, path_parts: List[str], output_dir: Path, 
        zip: bool = False, hide_missing: bool = False, include_original_name: bool = True, print_missing: bool = False,
        extension: Optional[str] = None, remove_if_exists: bool = True) -> List[str]:
    '''Collect results of the same type. Return the list of families with missing results.'''
    if output_dir.is_dir() and remove_if_exists:
        lib_sh.rm(output_dir, recursive=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for family in families:
        inp = Path(input_dir, 'families', family, *path_parts)
        filename = Path(path_parts[-1])
        if include_original_name:
            if extension is not None:
                ext = extension
                assert filename.name.endswith(ext)  # might be e.g. '.sses.json'
                name = filename.name[:-len(ext)]
            else:
                ext = filename.suffix
                name = filename.stem
            out = output_dir / f'{name}-{family}{ext}'
        else:
            out = output_dir / family
        if inp.exists():
            if zip:
                shutil.make_archive(str(out), 'zip', str(inp))
            else:
                lib_sh.cp(inp, out)
        elif hide_missing:
            missing.append(family)
            is_empty = (input_dir/'families'/family/'EMPTY_FAMILY').exists()
            error_message = f"Failed to generate consensus for '{family}'."
            if is_empty:
                error_message += " The family is empty."
            lib.dump_json({'error': error_message}, out)
        else:
            missing.append(family)
            if print_missing:
                print('Missing results/ directory:', family, file=sys.stderr)
    return missing

def _download_domains(outdir: Path, family: str) -> None:
    with RedirectIO(stdout=outdir/f'{family}.json', stderr=os.devnull):
        domains_from_pdbeapi.main(family, join_domains_in_chain=True)

def get_domain_lists(families: List[str], outdir: Path, collected_output_json: Optional[Path], collected_output_txt: Optional[Path] = None, 
                     processes: Optional[int] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Downloading domain lists for {len(families)} families:')
    jobs = [lib_multiprocessing.Job(name=family, func=_download_domains, args=(outdir, family)) for family in families]
    lib_multiprocessing.run_jobs_with_multiprocessing(jobs, n_processes=processes, progress_bar=True)
    if collected_output_json is not None or collected_output_txt is not None:
        collected = {}
        for family in families:
            with open(outdir/f'{family}.json') as r:
                js = json.load(r)
            collected[family] = js
    if collected_output_json is not None:
        lib.dump_json(collected, collected_output_json)
    if collected_output_txt is not None:
        domain_list = []
        for family, pdb2doms in collected.items():
            for pdb, doms in pdb2doms.items():
                for dom in doms:
                    domain_list.append((family, dom['domain'], dom['pdb'], dom['chain_id'], dom['ranges'] ))
        domain_list.sort()
        with open(collected_output_txt, 'w') as w:
            for domain in domain_list:
                print(*domain, sep='\t', file=w)


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family_list_file', help='File with list of family codes (whitespace-separated)', type=Path)
    parser.add_argument('sample_size', help='Number of domains to process per family (integer or "all")', type=str)
    parser.add_argument('directory', help='Directory to save everything in', type=Path)
    parser.add_argument('-d', '--download_family_list', help='Download the current list of all CATH families (ignore family_list_file)', action='store_true')
    parser.add_argument('-D', '--download_family_list_by_size', help='Same as -d, but sort the families by size (largest first)', action='store_true')
    parser.add_argument('--config', help=f'Configuration file for OverProt', type=Path, default=None)
    parser.add_argument('--collect', help='Collect result files of specific types (diagram.json, consensus.png, results.zip) in directory/collected_resuts/', action='store_true')
    parser.add_argument('--only_get_lists', help='Get domain lists for families and exit', action='store_true')
    parser.add_argument('--processes', help='Number of processes to run (default: number of CPUs)', type=int)
    parser.add_argument('--out', help='File for stdout.', type=Path)
    parser.add_argument('--err', help='File for stderr.', type=Path)
    args = parser.parse_args()
    return vars(args)


def main(family_list_file: Path, sample_size: int|str|None, directory: Path, 
         download_family_list: bool = False, download_family_list_by_size: bool = False, collect: bool = False, config: Optional[Path] = None,
         only_get_lists: bool = False, processes: Optional[int] = None, out: Optional[Path] = None, err: Optional[Path] = None) -> Optional[int]:
    with RedirectIO(stdout=out, stderr=err), Timing('Total'):
        start_time = datetime.now().astimezone()
        print('Output directory:', directory)
        directory.mkdir(parents=True, exist_ok=True)
        if download_family_list_by_size:
            family_list_file = directory/'families.txt'
            get_cath_family_list.main(directory/'cath-domain-list.txt', download=True, output=family_list_file, sort_by_size=True)
        elif download_family_list:
            family_list_file = directory/'families.txt'
            get_cath_family_list.main(directory/'cath-domain-list.txt', download=True, output=family_list_file)
        print('Family list file:', family_list_file)
        text = family_list_file.read_text()
        families = text.split()
        print('Number of families:', len(families))
        get_domain_lists(families, directory/'domain_lists', collected_output_json=directory/'domain_list.json', collected_output_txt=directory/'domain_list.txt', processes=processes)
        get_cath_example_domains.main(output=directory/'cath_example_domains.txt')
        get_cath_family_names.main(directory/'cath-b-newest-names.gz', download=True, output=directory/'cath_b_names_options.json')
        if only_get_lists:
            return None
        out_err_dir = directory/'stdout_stderr'
        out_err_dir.mkdir(exist_ok=True)
        current_dir = directory/'current'
        current_dir.mkdir(exist_ok=True)
        (directory/'families.txt').write_text('\n'.join(families))
        (directory/'failed_families.txt').write_text('')
        (directory/'succeeded_families.txt').write_text('')
        (directory/'errors.txt').write_text('')
        jobs = [
            lib_multiprocessing.Job(
                name=family, 
                func=process_family, 
                args=(family, sample_size, directory/'families'/family),
                kwargs={'config': config, 'domains': directory/'domain_lists'/f'{family}.json'}, 
                stdout=current_dir/f'{family}-out.txt',
                stderr=current_dir/f'{family}-err.txt'
            ) for family in families]
        results = lib_multiprocessing.run_jobs_with_multiprocessing(jobs, n_processes=processes, progress_bar=True, 
            callback = lambda res: family_callback(res, directory))
        lib_sh.rm(current_dir, recursive=True, ignore_errors=True)
        if collect:
            collect_results(families, directory, ['results'], directory/'collected_results'/'zip_results', zip=True)
            collect_results(families, directory, ['results', 'diagram.json'], directory/'collected_results'/'diagrams', hide_missing=True)
            collect_results(families, directory, ['lists'], directory/'collected_results'/'families', include_original_name=False)
            collect_results(families, directory, ['results', 'consensus.cif'], directory/'collected_results'/'consensus')
            collect_results(families, directory, ['results', 'consensus.sses.json'], directory/'collected_results'/'consensus', remove_if_exists=False, extension='.sses.json')
            bulk_dir = directory/'collected_results'/'bulk'
            bulk_dir.mkdir(parents=True, exist_ok=True)
            lib_sh.archive(directory/'collected_results'/'consensus', bulk_dir/'consensus.zip')
            missing_families = collect_results(families, directory, ['results', 'consensus.png'], directory/'collected_results'/'consensus_3d', print_missing=True)
            with open(directory/'missing_results.txt', 'w') as w:
                for family in missing_families:
                    print(family, file=w)
            lib_sh.cp(directory/'domain_list.json', directory/'collected_results')
            lib_sh.cp(directory/'domain_list.txt', directory/'collected_results')
            lib_sh.cp(directory/'cath_example_domains.txt', directory/'collected_results')
            lib_sh.cp(directory/'cath_b_names_options.json', directory/'collected_results')
            (directory/'collected_results'/'last_update.txt').write_text(start_time.strftime('%d %B %Y').lstrip('0'))
            (directory/'collected_results'/'last_update_iso.txt').write_text(start_time.isoformat(timespec='seconds'))
        succeeded = sum(1 for res in results if res.result is None)
        failed = sum(1 for res in results if res.result is not None)
        print('Succeeded:', succeeded, 'Failed:', failed)
        return None

def _main():
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)

if __name__ == '__main__':
    _main()
