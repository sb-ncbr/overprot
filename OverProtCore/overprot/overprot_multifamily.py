'''
Run OverProt algorithm for multiple families in parallel processes.

Example usage:
    python3  -m overprot.overprot_multifamily  -  data/  --download_family_list
'''

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, List, Iterable, Callable
import traceback
import contextlib
import json
from datetime import datetime

from .libs import lib
from .libs import lib_sh
from .libs import lib_multiprocessing
from .libs.lib_io import RedirectIO
from .libs.lib_logging import Timing
from .libs.lib_dependencies import DEFAULT_CONFIG_FILE
from .libs.lib_overprot_config import OverProtConfig, ConfigException
from . import get_cath_family_list
from . import get_cath_example_domains
from . import get_cath_family_names
from . import get_pdb_entry_list
from . import make_chain_summaries
from . import domains_from_pdbeapi
from . import overprot
from .libs.lib_cli import cli_command, run_cli_command


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

def collect_results(xs: Iterable[str], in_dir: Path, in_subpath: str, out_dir: Path, out_subpath: str, 
        alternative_content: Callable[[str], str]|None = None, print_missing: bool = False, 
        breakdown_function: Callable[[Path], Path|str]|None = None) -> List[str]:
    '''Collect results of the same type. Return the list of families with missing results.'''
    missing = []
    for x in xs:
        in_file = in_dir / in_subpath.format(x=x)
        out_file = out_dir / out_subpath.format(x=x)
        glob = '*' in in_file.name
        zip = out_file.suffix == '.zip'
        if glob:
            files = sorted(in_file.parent.glob(in_file.name))
            tmp_out = out_file.with_suffix('.tmp') if zip else out_file
            tmp_out.mkdir(parents=True, exist_ok = not zip)
            if breakdown_function is None:
                for file in files:
                    lib_sh.cp(file, tmp_out/file.name)
            else:
                for file in files:
                    lib_sh.cp(file, tmp_out/breakdown_function(file)/file.name)
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

def download_domains(outdir: Path, family: str) -> None:
    with RedirectIO(stdout=outdir/f'{family}.json', stderr=os.devnull):
        domains_from_pdbeapi.main(family, join_domains_in_chain=True)

def get_domain_lists(families: List[str], outdir: Path, collected_output_json: Optional[Path], collected_output_csv: Optional[Path] = None, 
                     processes: Optional[int] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Downloading domain lists for {len(families)} families:')
    jobs = [lib_multiprocessing.Job(name=family, func=download_domains, args=(outdir, family)) for family in families]
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

@cli_command(parsers={'sample_size': lib.int_or_all}, short_options={'download_family_list': '-d', 'download_family_list_by_size': '-D'})
def main(family_list_file: Path, directory: Path, sample_size: Optional[int] = None, 
         download_family_list: bool = False, download_family_list_by_size: bool = False, collect: bool = False, config: Optional[Path] = None,
         only_get_lists: bool = False, extras: bool = False, processes: Optional[int] = None, out: Optional[Path] = None, err: Optional[Path] = None) -> None:
    '''Run OverProt algorithm for multiple families in parallel processes.
    @param  `family_list_file`  File with list of family codes (whitespace-separated).
    @param  `directory`         Directory for results.
    @param  `sample_size`       Number of domains to process per family (integer or "all"). [default: "all"]
    @param  `download_family_list`          Download the current list of all CATH families (ignore family_list_file).
    @param  `download_family_list_by_size`  Same as --download_family_list, but sort the families by size (largest first).
    @param  `config`            Configuration file for OverProt.
    @param  `collect`           Collect result files of specific types (diagram.json, consensus.png, results.zip) in directory/collected_resuts/.
    @param  `only_get_lists`    Get domain lists for families and exit.
    @param  `extras`            Get some extra information (CATH example domains, family name, PDB entry list, PDB chain summaries).
    @param  `processes`         Number of processes to run. [default: number of CPUs]
    @param  `out`               File for stdout.
    @param  `err`               File for stderr.
    '''
    with RedirectIO(stdout=out, stderr=err), Timing('Total'):
        if config is None:
            config = DEFAULT_CONFIG_FILE
        try:
            conf = OverProtConfig(config)
        except OSError:
            print(f'ERROR: Cannot open configuration file: {config}', file=sys.stderr)
            return 1
        except ConfigException as ex:
            print('ERROR:', ex, file=sys.stderr)
            return 2
        start_time = datetime.now().astimezone()
        print('Output directory:', directory)
        directory.mkdir(parents=True, exist_ok=True)
        if download_family_list_by_size:
            family_list_file = directory/'families.txt'
            get_cath_family_list.main(directory/'cath-superfamily-list.txt', download=True, out=family_list_file, sort_by_size=True)
        elif download_family_list:
            family_list_file = directory/'families.txt'
            get_cath_family_list.main(directory/'cath-superfamily-list.txt', download=True, out=family_list_file)
        print('Family list file:', family_list_file)
        text = family_list_file.read_text()
        families = text.split()
        print('Number of families:', len(families))
        get_domain_lists(families, directory/'domain_lists', collected_output_json=directory/'domain_list.json', collected_output_csv=directory/'domain_list.csv', processes=processes)
        if extras:
            get_cath_example_domains.main(output=directory/'cath_example_domains.csv')
            get_cath_family_names.main(directory/'cath-b-newest-names.gz', download=True, output=directory/'cath_b_names_options.json')
            get_pdb_entry_list.main(out=directory/'pdbs.txt')
            make_chain_summaries.main(directory/'pdbs.txt', directory/'chain_summaries', conf.download.structure_sources, breakdown=True, processes=processes)
        if only_get_lists:
            return
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
                collect_results(families, f, '{x}/lists/family_info.txt',                c, 'family/info/family_info-{x}.txt')
                collect_results(families, f, '{x}/lists/*',                              c, 'family/lists/{x}/')
                collect_results(families, f, '{x}/results/',                             c, 'family/zip_results/results-{x}.zip')
                collect_results(families, f, '{x}/results/diagram.json',                 c, 'family/diagram/diagram-{x}.json', alternative_content = lambda fam: failed_diagram_content(fam, (f/fam/'EMPTY_FAMILY').exists()))
                collect_results(families, f, '{x}/results/consensus.cif',                c, 'family/consensus_cif/consensus-{x}.cif')
                collect_results(families, f, '{x}/results/consensus.sses.json',          c, 'family/consensus_sses/consensus-{x}.sses.json')
                collect_results(families, f, '{x}/results/consensus.png',                c, 'family/consensus_3d/consensus-{x}.png')
                collect_results(families, f, '{x}/annotated_sses/*-annotated.sses.json', c, 'family/annotation/annotation-{x}.zip')
                collect_results(families, f, '{x}/annotated_sses/*-annotated.sses.json', c, 'domain/annotation/', breakdown_function = lambda file: file.name[1:3])
                collect_results(families, f, '{x}/domain_info/*.json',                   c, 'domain/info/', breakdown_function = lambda file: file.name[1:3])
                collect_results(families, f, '{x}/cif_cealign/*-rotation.json',          c, 'domain/rotation/', breakdown_function = lambda file: file.name[1:3])
                
                lib_sh.cp(directory/'families.txt', c)
                lib_sh.cp(directory/'domain_list.json', c)
                lib_sh.cp(directory/'domain_list.csv', c)

                lib_sh.archive(c/'family'/'consensus_cif',  c/'bulk'/'family'/'consensus_cif.zip')
                lib_sh.archive(c/'family'/'consensus_sses', c/'bulk'/'family'/'consensus_sses.zip')
                lib_sh.archive(c/'domain'/'rotation',       c/'bulk'/'domain'/'rotation.zip')
                
                if extras:
                    lib_sh.cp(directory/'pdbs.txt', c)
                    lib_sh.cp(directory/'cath_example_domains.csv', c)
                    lib_sh.cp(directory/'cath_b_names_options.json', c)
                    lib_sh.cp(directory/'chain_summaries', c/'pdb'/'chain_summary')
                    lib_sh.archive(c/'pdb'/'chain_summary', c/'bulk'/'pdb'/'chain_summary.zip')

                (c/'last_update.txt').write_text(start_time.strftime('%d %B %Y').lstrip('0'))
                (c/'last_update_iso.txt').write_text(start_time.isoformat(timespec='seconds'))

                missing = [family for family in families if not (c/'family'/'consensus_3d'/f'consensus-{family}.png').is_file()]
                (directory/'missing_results.txt').write_text('\n'.join(missing))


def _main():
    run_cli_command(main)


if __name__ == '__main__':
    _main()
