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
# from . import collect_diagrams

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################

HLINE = '-' * 60

def process_family(family: str, sample_size: Union[int, str, None], directory: FilePath) -> Optional[str]:
    '''Try running SecStrConsensus on a family. Return the error traceback if fails, None if succeeds.'''
    print(HLINE, family, sep='\n')
    print(HLINE, family, sep='\n', file=sys.stderr)
    try:
        overprot.main(family, sample_size, directory)
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



def collect_results(families: List[str], input_dir: FilePath, filepath: List[str], output_dir: FilePath, zip: bool = False, hide_missing: bool = False):
    if output_dir.isdir():
        output_dir.rm(recursive=True)
    output_dir.mkdir()
    for family in families:
        inp = input_dir.sub('families', family, *filepath)
        filename = FilePath(filepath[-1])
        out = output_dir.sub(f'{filename.name}-{family}{filename.ext}')
        if inp.exists():
            if zip:
                shutil.make_archive(str(out), 'zip', str(inp))
            else:
                inp.cp(out)
        elif hide_missing:
            with out.open('w') as w:
                js_error = {'error': f"Failed to generate consensus for '{family}'"}
                json.dump(js_error, w)
        else:
            print('Missing results/ directory:', family, file=sys.stderr)

    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    # if png_dir is not None:
    #     Path(png_dir).mkdir(parents=True, exist_ok=True)

    # fam_dirs = sorted(fam for fam in Path(input_dir, 'families').iterdir() if fam.is_dir)
    # # print(fam_dirs)
    # for fd in fam_dirs:
    #     family = fd.name
    #     try:
    #         make_archive(Path(fd, 'results'), Path(output_dir, f'results-{family}.zip'))
    #         shutil.copy(Path(fd, 'results', 'consensus.png'), Path(png_dir, f'{family}.png'))
    #     except FileNotFoundError:
    #         print('Missing results/ directory:', family, file=sys.stderr)
    



#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family_list_file', help='File with list of family codes (whitespace-separated)', type=str)
    parser.add_argument('sample_size', help='Number of domains to process per family (integer or "all")', type=str)
    parser.add_argument('directory', help='Directory to save everything in', type=str)
    parser.add_argument('-d', '--download_family_list', help='Download the current list of all CATH families (ignore family_list_file)', action='store_true')
    parser.add_argument('-D', '--download_family_list_by_size', help='Same as -d, but sort the families by size (largest first)', action='store_true')
    parser.add_argument('-c', '--collect', help='Collect result files of specific types (diagram.json, consensus.png, results.zip) in directory/collected_resuts/', action='store_true')
    args = parser.parse_args()
    return vars(args)


def main(family_list_file: Union[FilePath, str], sample_size: Union[int, str, None], directory: Union[FilePath, str], 
         download_family_list: bool = False, download_family_list_by_size: bool = False, collect: bool = False) -> Optional[int]:
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
            kwargs={}, 
            stdout=current_dir.sub(f'{family}-out.txt'),
            stderr=current_dir.sub(f'{family}-err.txt')
        ) for family in families]
    results = lib.run_jobs_with_multiprocessing(jobs, n_processes=None, progress_bar=True, 
        callback = lambda res: family_callback(res, FilePath(directory)))
    current_dir.rm(recursive=True, ignore_errors=True)
    if collect:
        collect_results(families, directory, ['results', 'diagram.json'], directory.sub('collected_results', 'diagrams'), hide_missing=True)
        collect_results(families, directory, ['results', 'consensus.png'], directory.sub('collected_results', 'consensus_3d'))
        collect_results(families, directory, ['results'], directory.sub('collected_results', 'zip_results'), zip=True)
        # collect_diagrams.main(directory, directory.sub('diagrams'))
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
