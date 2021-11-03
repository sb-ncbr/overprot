'''
This Python3 script does foo ...

Example usage:
    python3  foo.py  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

import sys
import argparse
from typing import Dict, Any, Optional, Union, List
import traceback
import contextlib
import json
import shutil

from .libs import lib
from .libs.lib import FilePath
from . import get_cath_family_list
from . import overprot

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################

HLINE = '-' * 60

def process_family(family: str, sample_size: Union[int, str, None], directory: FilePath, config: Optional[FilePath] = None) -> Optional[str]:
    '''Try running OverProt on a family. Return the error traceback if fails, None if succeeds.'''
    print(HLINE, family, sep='\n')
    print(HLINE, family, sep='\n', file=sys.stderr)
    try:
        overprot.main(family, sample_size, directory, config=config)
        return None
    except Exception as ex:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        return error

def family_callback(result: lib.JobResult, directory: FilePath):
    family = result.job.name
    error = result.result
    with contextlib.suppress(OSError):
        directory.sub('current', f'{family}-out.txt').cp(directory.sub('families', family, 'stdout.txt'))
        directory.sub('current', f'{family}-err.txt').cp(directory.sub('families', family, 'stderr.txt'))
        directory.sub('current', f'{family}-out.txt').mv(directory.sub('stdout_stderr', f'{family}-out.txt'))
        directory.sub('current', f'{family}-err.txt').mv(directory.sub('stdout_stderr', f'{family}-err.txt'))
    if error is not None:
        with directory.sub('failed_families.txt').open('a') as w:
            print(family, file=w)
        with directory.sub('errors.txt').open('a') as w:
            print(family, error, sep='\n', end='\n\n', file=w)
    else:
        with directory.sub('succeeded_families.txt').open('a') as w:
            print(family, file=w)



def collect_results(families: List[str], input_dir: FilePath, filepath: List[str], output_dir: FilePath, 
        zip: bool = False, hide_missing: bool = False, include_original_name: bool = True, print_missing: bool = False,
        extension: Optional[str] = None, remove_if_exists: bool = True) -> List[str]:
    '''Collect results of the same type. Return the list of families with missing results.'''
    if output_dir.isdir() and remove_if_exists:
        output_dir.rm(recursive=True)
    output_dir.mkdir(exist_ok=True)
    missing = []
    for family in families:
        inp = input_dir.sub('families', family, *filepath)
        filename = FilePath(filepath[-1])
        if include_original_name:
            if extension is not None:
                ext = extension
                name = filename.base[:-len(ext)]
            else:
                ext = filename.ext
                name = filename.name
            out = output_dir.sub(f'{name}-{family}{ext}') 
        else:
            out = output_dir.sub(family)
        if inp.exists():
            if zip:
                shutil.make_archive(str(out), 'zip', str(inp))
            else:
                inp.cp(out)
        elif hide_missing:
            missing.append(family)
            is_empty = input_dir.sub('families', family, 'EMPTY_FAMILY').exists()
            error_message = f"Failed to generate consensus for '{family}'."
            if is_empty:
                error_message += " The family is empty."
            out.dump_json({'error': error_message})
        else:
            missing.append(family)
            if print_missing:
                print('Missing results/ directory:', family, file=sys.stderr)
    return missing


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family_list_file', help='File with list of family codes (whitespace-separated)', type=str)
    parser.add_argument('sample_size', help='Number of domains to process per family (integer or "all")', type=str)
    parser.add_argument('directory', help='Directory to save everything in', type=str)
    parser.add_argument('-d', '--download_family_list', help='Download the current list of all CATH families (ignore family_list_file)', action='store_true')
    parser.add_argument('-D', '--download_family_list_by_size', help='Same as -d, but sort the families by size (largest first)', action='store_true')
    parser.add_argument('--config', help=f'Configuration file for OverProt', type=str, default=None)
    parser.add_argument('--collect', help='Collect result files of specific types (diagram.json, consensus.png, results.zip) in directory/collected_resuts/', action='store_true')
    args = parser.parse_args()
    return vars(args)


def main(family_list_file: Union[FilePath, str], sample_size: Union[int, str, None], directory: Union[FilePath, str], 
         download_family_list: bool = False, download_family_list_by_size: bool = False, collect: bool = False, config: Union[FilePath, str, None] = None) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    directory = FilePath(directory)
    print('Output directory:', directory)
    directory.mkdir(exist_ok=True)
    if download_family_list_by_size:
        family_list_file = directory.sub('families.txt')
        get_cath_family_list.main(directory.sub('cath-domain-list.txt'), download=True, output=family_list_file, sort_by_size=True)
    elif download_family_list:
        family_list_file = directory.sub('families.txt')
        get_cath_family_list.main(directory.sub('cath-domain-list.txt'), download=True, output=family_list_file)
    else:
        family_list_file = FilePath(family_list_file)
    print('Family list file:', family_list_file)
    with family_list_file.open() as f:
        text = f.read()
    families = text.split()
    print('Number of families:', len(families))
    out_err_dir = directory.sub('stdout_stderr').mkdir(exist_ok=True)
    current_dir = directory.sub('current').mkdir(exist_ok=True)
    with directory.sub('families.txt').open('w') as w:
        w.write('\n'.join(families))
    directory.sub('failed_families.txt').clear()
    directory.sub('succeeded_families.txt').clear()
    directory.sub('errors.txt').clear()
    jobs = [
        lib.Job(
            name=family, 
            func=process_family, 
            args=(family, sample_size, directory.sub('families', family)), 
            kwargs={'config': config}, 
            stdout=current_dir.sub(f'{family}-out.txt'),
            stderr=current_dir.sub(f'{family}-err.txt')
        ) for family in families]
    results = lib.run_jobs_with_multiprocessing(jobs, n_processes=None, progress_bar=True, 
        callback = lambda res: family_callback(res, FilePath(directory)))
    current_dir.rm(recursive=True, ignore_errors=True)
    if collect:
        collect_results(families, directory, ['results'], directory.sub('collected_results', 'zip_results'), zip=True)
        collect_results(families, directory, ['results', 'diagram.json'], directory.sub('collected_results', 'diagrams'), hide_missing=True)
        collect_results(families, directory, ['lists'], directory.sub('collected_results', 'families'), include_original_name=False)
        collect_results(families, directory, ['results', 'consensus.cif'], directory.sub('collected_results', 'consensus'))
        collect_results(families, directory, ['results', 'consensus.sses.json'], directory.sub('collected_results', 'consensus'), remove_if_exists=False, extension='.sses.json')
        bulk_dir = directory.sub('collected_results', 'bulk').mkdir()
        shutil.make_archive(str(bulk_dir.sub('consensus')), 'zip', str(directory.sub('collected_results', 'consensus')))
        missing_families = collect_results(families, directory, ['results', 'consensus.png'], directory.sub('collected_results', 'consensus_3d'), print_missing=True)
        with directory.sub('missing_results.txt').open('w') as w:
            for family in missing_families:
                print(family, file=w)
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
