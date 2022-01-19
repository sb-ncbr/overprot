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
from typing import Dict, Any, Optional, List, Iterable, Callable
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

def process_family(family: str, directory: Path, sample_size: int|str|None = None, config: Optional[Path] = None, domains: Optional[Path] = None) -> Optional[str]:
    '''Try running OverProt on a family. Return the error traceback if fails, None if succeeds.'''
    print(HLINE, family, sep='\n')
    print(HLINE, family, sep='\n', file=sys.stderr)
    try:
        overprot.main(family, directory, sample_size=sample_size, config=config, domains=domains)
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

def failed_diagram_content(family: str, is_empty: bool) -> str:
    error_message = f"Failed to generate consensus for '{family}'."
    if is_empty:
        error_message += " The family is empty."
    return json.dumps({'error': error_message})

def collect_results2(xs: Iterable[str], in_dir: Path, in_subpath: str, out_dir: Path, out_subpath: str, 
        alternative_content: Callable[[str], str]|None = None, print_missing: bool = False, 
        breakout_function: Callable[[Path], Path|str]|None = None) -> List[str]:
    '''Collect results of the same type. Return the list of families with missing results.'''
    missing = []
    for x in xs:
        in_file = in_dir / in_subpath.format(x=x)
        out_file = out_dir / out_subpath.format(x=x)
        # is_dir = in_file.is_dir()
        glob = '*' in in_file.name
        zip = out_file.suffix == '.zip'
        if glob:
            files = sorted(in_file.parent.glob(in_file.name))
            tmp_out = out_file.with_suffix('.tmp') if zip else out_file
            tmp_out.mkdir(parents=True, exist_ok = not zip)
            if breakout_function is None:
                for file in files:
                    lib_sh.cp(file, tmp_out/file.name)
            else:
                for file in files:
                    lib_sh.cp(file, tmp_out/breakout_function(file)/file.name)
            if zip:
                lib_sh.archive(tmp_out, out_file, rm_source=True)
        else:
            if in_file.exists():
                if zip: 
                    lib_sh.archive(in_file, out_file)
                else:
                    lib_sh.cp(in_file, out_file)
            else:
                missing.append(x)
                if alternative_content is not None:
                    out_file.write_text(alternative_content(x))
                if print_missing:
                    print('Missing results:', in_subpath.format(x=x), file=sys.stderr)
    return missing


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

def get_domain_lists(families: List[str], outdir: Path, collected_output_json: Optional[Path], collected_output_csv: Optional[Path] = None, 
                     processes: Optional[int] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Downloading domain lists for {len(families)} families:')
    jobs = [lib_multiprocessing.Job(name=family, func=_download_domains, args=(outdir, family)) for family in families]
    lib_multiprocessing.run_jobs_with_multiprocessing(jobs, n_processes=processes, progress_bar=True)
    if collected_output_json is not None or collected_output_csv is not None:
        collected = {}
        for family in families:
            with open(outdir/f'{family}.json') as r:
                js = json.load(r)
            collected[family] = js
    if collected_output_json is not None:
        lib.dump_json(collected, collected_output_json)
    if collected_output_csv is not None:
        domain_list = []
        for family, pdb2doms in collected.items():
            for pdb, doms in pdb2doms.items():
                for dom in doms:
                    domain_list.append((family, dom['domain'], dom['pdb'], dom['chain_id'], dom['ranges'] ))
        domain_list.sort()
        with open(collected_output_csv, 'w') as w:
            print('family;domain;pdb;chain_id;ranges', file=w)
            for domain in domain_list:
                print(*domain, sep=';', file=w)


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family_list_file', help='File with list of family codes (whitespace-separated)', type=Path)
    parser.add_argument('directory', help='Directory to save everything in', type=Path)
    parser.add_argument('--sample_size', help='Number of domains to process per family (integer or "all")', type=str, default='all')
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


def main(family_list_file: Path, directory: Path, sample_size: int|str|None = None, 
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
        get_domain_lists(families, directory/'domain_lists', collected_output_json=directory/'domain_list.json', collected_output_csv=directory/'domain_list.csv', processes=processes)
        get_cath_example_domains.main(output=directory/'cath_example_domains.csv')
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
                args=(family, directory/'families'/family),
                kwargs={'sample_size': sample_size, 'config': config, 'domains': directory/'domain_lists'/f'{family}.json'}, 
                stdout=current_dir/f'{family}-out.txt',
                stderr=current_dir/f'{family}-err.txt'
            ) for family in families]
        print(f'Running OverProt for {len(families)} families:')
        results = lib_multiprocessing.run_jobs_with_multiprocessing(jobs, n_processes=processes, progress_bar=True, 
            callback = lambda res: family_callback(res, directory))
        lib_sh.rm(current_dir, recursive=True, ignore_errors=True)
        succeeded = sum(1 for res in results if res.result is None)
        failed = sum(1 for res in results if res.result is not None)
        print('Succeeded:', succeeded, 'Failed:', failed)
        if collect:
            print('Collecting results:')
            with Timing('Collecting results'):
                f = directory/'families'
                c = directory/'collected_results'
                collect_results2(families, f, '{x}/lists/family_info.txt',                c, 'family/info/family_info-{x}.txt')
                collect_results2(families, f, '{x}/lists/*',                              c, 'family/lists/{x}/')
                collect_results2(families, f, '{x}/results/',                             c, 'family/zip_results/results-{x}.zip')
                collect_results2(families, f, '{x}/results/diagram.json',                 c, 'family/diagram/diagram-{x}.json', alternative_content = lambda fam: failed_diagram_content(fam, (f/fam/'EMPTY_FAMILY').exists()))
                collect_results2(families, f, '{x}/results/consensus.cif',                c, 'family/consensus_cif/consensus-{x}.cif')
                collect_results2(families, f, '{x}/results/consensus.sses.json',          c, 'family/consensus_sses/consensus-{x}.sses.json')
                collect_results2(families, f, '{x}/results/consensus.png',                c, 'family/consensus_3d/consensus-{x}.png')
                collect_results2(families, f, '{x}/annotated_sses/*-annotated.sses.json', c, 'family/annotation/annotation-{x}.zip')
                collect_results2(families, f, '{x}/annotated_sses/*-annotated.sses.json', c, 'domain/annotation/', breakout_function = lambda file: file.name[1:3])
                lib_sh.archive(c/'family'/'consensus_cif',  c/'bulk'/'family'/'consensus_cif.zip')
                lib_sh.archive(c/'family'/'consensus_sses', c/'bulk'/'family'/'consensus_sses.zip')
                
                lib_sh.cp(directory/'families.txt', c)
                lib_sh.cp(directory/'domain_list.json', c)
                lib_sh.cp(directory/'domain_list.csv', c)
                lib_sh.cp(directory/'cath_example_domains.csv', c)
                lib_sh.cp(directory/'cath_b_names_options.json', c)
                (c/'last_update.txt').write_text(start_time.strftime('%d %B %Y').lstrip('0'))
                (c/'last_update_iso.txt').write_text(start_time.isoformat(timespec='seconds'))

                missing = [family for family in families if not (c/'family'/'consensus_3d'/f'consensus-{family}.png').is_file()]
                (directory/'missing_results.txt').write_text('\n'.join(missing))
        return None

def _main():
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)

if __name__ == '__main__':
    _main()
