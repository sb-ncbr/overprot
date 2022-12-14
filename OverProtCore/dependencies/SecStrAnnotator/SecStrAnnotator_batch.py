'''
This Python3 script runs SecStrAnnotator on multiple query protein domains in one batch.

Example usage:
    python3  SecStrAnnotator_batch.py  --help
    python3  SecStrAnnotator_batch.py  examples/  2nnj,A  examples/cyp_family_sample.json - -threads 4
'''

import argparse
from typing import Dict, Any, Optional, Union, List
import json
import os
from os import path
import shutil
import sys
from collections import OrderedDict
import queue
import threading
import subprocess

#  CONSTANTS  ################################################################################

DEFAULT_SECSTRANNOTATOR_DLL = path.join(path.dirname(__file__), 'SecStrAnnotator.dll')

#  FUNCTIONS  ################################################################################

def clear_file(filename):
    with open(filename, 'w') as w:
        w.write('')

def try_rename_file(source, dest):
    try:
        os.rename(source, dest)
    except:
        pass

def copy_file(source, dest, append=False):
    with open(source, 'r') as r:
        with open(dest, 'a' if append else 'w') as w:
            for line in iter(r.readline, ''):
                w.write(line)

def remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

def try_read_json(filename):
    with open(filename) as f:
        try:
            result = json.load(f, object_pairs_hook=OrderedDict)
        except ValueError as e:
            raise Exception(f'File "{filename}" is not a valid JSON file ({e}) \n')
    return result

def check_json_type(json_object, typeex):
    if isinstance(typeex, tuple):
        return any( check_json_type(json_object, typ) for typ in typeex )
    elif isinstance(typeex, str):
        return isinstance(json_object, str)
    elif isinstance(typeex, bool):
        return isinstance(json_object, bool)
    elif isinstance(json_object, (int, float)):
        return isinstance(json_object, (int, float))
    elif typeex is None:
        return json_object is None
    elif isinstance(typeex, list):
        if not isinstance(json_object, list):
            return False 
        if len(typeex) > 0:
            return all( check_json_type(elem, typeex[0]) for elem in json_object )
        else: 
            return True
    elif isinstance(typeex, dict):
        if not isinstance(json_object, dict):
            return False 
        if len(typeex) > 0:
            value_typeex = next(iter(typeex.values()))
            return all( isinstance(key, str) and check_json_type(value, value_typeex) for key, value in json_object.items() )
        else: 
            return True
    raise Exception('Invalid typeex')

def read_domains_json(filename):
    result = try_read_json(filename)
    if check_json_type(result, {'PDB':[['']]}):
        return result
    elif check_json_type(result, {'PDB':[{}]}):
        return convert_dict_domains_to_tuple_domains(result)
    else:
        raise Exception(f'Expected a JSON format {{PDB:[[domain_name,chain,ranges]]}} in "{filename}" \n')

def convert_dict_domains_to_tuple_domains(js: Dict[str, List[dict]]) -> Dict[str, List[List[str]]]:
    '''Reformat json object from {PDB:[{"domain": domain_name, "chain_id": chain, "ranges": ranges}]} to {PDB:[[domain_name,chain,ranges]]}'''
    result = {}
    for pdb, domains in js.items():
        simple_domains = []
        for i, domain in enumerate(domains):
            try:
                simple_domains.append((domain['domain'], domain['chain_id'], domain['ranges']))
            except KeyError as err:
                missing_key = err.args[0]
                raise Exception(f"Wrong input format: missing key '{missing_key}' in PDB {pdb}, domain #{i+1} ({dict(domain)})")
        result[pdb] = simple_domains
    return result

def run_in_threads (do_job, jobs, n_threads, progress_bar=None, initialize_thread_sync=None, finalize_thread_sync=None):
    q = queue.Queue()
    for job in jobs:
        q.put(job)
    def worker():
        all_done = False
        while not all_done:
            try:
                job = q.get(block=False)
                do_job(job)
                if progress_bar is not None:
                    progress_bar.step()
                q.task_done()
            except queue.Empty:
                all_done = True
    threads = [ threading.Thread(target=worker) for i in range(n_threads) ]
    if progress_bar is not None:
        progress_bar.start()
    if initialize_thread_sync is not None:
        for thread in threads:
            initialize_thread_sync(thread)
    for thread in threads:
        thread.start()
    q.join()
    if finalize_thread_sync is not None:
        for thread in threads:
            finalize_thread_sync(thread)
    if progress_bar is not None:
        progress_bar.finalize()

class ProgressBar:
    DONE_SYMBOL = '???'
    TODO_SYMBOL = '-'
    def __init__(self, n_steps, width=None, title='', prefix='', suffix='', writer=None):
        self.n_steps = n_steps # expected number of steps
        self.width = width if width is not None else shutil.get_terminal_size().columns
        self.width -= len(prefix) + len(suffix) + 12
        self.title = (' '+title+' ')[0:min(len(title)+2, self.width)]
        self.prefix = prefix
        self.suffix = suffix
        self.writer = writer if writer is not None else sys.stdout
        self.done = 0 # number of completed steps
        self.shown = 0 # number of shown symbols

    def start(self):
        self.writer.write(' ' * len(self.prefix))
        self.writer.write(' ???' + self.title + '???'*(self.width-len(self.title)) + '???\n')
        self.writer.flush()
        self.step(0, force=True)
        return self

    def step(self, n_steps=1, force=False):
        if n_steps == 0 and not force:
            return
        self.done = min(self.done + n_steps, self.n_steps)
        progress = self.done / self.n_steps if self.n_steps > 0 else 1.0
        new_shown = int(self.width * progress)
        if new_shown != self.shown or force:
            self.writer.write(f'\r{self.prefix} ???')
            self.writer.write(self.DONE_SYMBOL * new_shown + self.TODO_SYMBOL * (self.width - new_shown))
            self.writer.write(f'??? {int(100*progress):>3}% {self.suffix} ')
            self.writer.flush()
            self.shown = new_shown  

    def finalize(self):
        self.step(self.n_steps - self.done)
        self.writer.write('\n')
        self.writer.flush()

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=str, help='directory with the input and output files (argument DIRECTORY for SecStrAnnotator)')
    parser.add_argument('template', type=str, help='template domain specification (argument TEMPLATE for SecStrAnnotator)')
    parser.add_argument('queries_file', type=str, help='JSON file with the list of domains to be annotated (in format {PDB:[[domain_name,chain,ranges]]}, will be processed to QUERY arguments for SecStrAnnotator. The alternative format is {PDB:[{"domain": domain_name, "chain_id": chain, "ranges": ranges}]})')
    parser.add_argument('--options', type=str, default='', help="Any options that are to be passed to SecStrAnnotator (must be enclosed in quotes and contain spaces, not to be confused with Python arguments, e.g. --options '--ssa dssp --soft' or ' --soft')")
    parser.add_argument('--by_pdb', action='store_true', help='Process queries PDB-by-PDB, ignore domain information (usable only with --options " --onlyssa")')
    parser.add_argument('--files_by_domain_name', action='store_true', help='Input files will be <domain_name>.cif instead of <PDB>.cif')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads (default: 1)')
    parser.add_argument('--dll', type=str, default=DEFAULT_SECSTRANNOTATOR_DLL, help=f'Path to the SecStrAnnotator DLL (default: {DEFAULT_SECSTRANNOTATOR_DLL})')
    args = parser.parse_args()
    return vars(args)


def main(directory: str, template: str, queries_file: str, options: Union[str, List[str]] = '', 
        by_pdb: bool = False, files_by_domain_name: bool = False, threads: int = 1, dll: str = DEFAULT_SECSTRANNOTATOR_DLL) -> Optional[int]:
    '''Run SecStrAnnotator on multiple query protein domains.'''

    if by_pdb and files_by_domain_name:
        raise Exception('You cannot set --by_pdb and --files_by_domain_name simultaneously')

    if isinstance(options, str):
        options = options.split()
    onlyssa = '--onlyssa' in options

    all_annotations_file = path.join(directory, 'all_annotations.sses.json')
    output = path.join(directory, 'stdout.txt')
    output_err = path.join(directory, 'stderr.txt')
    out_files_extensions = ['-aligned.cif', '-alignment.json', '-detected.sses.json', '-annotated.sses.json', '-annotated.pse']

    # Determine whether can run dotnet SecStrAnnotator.dll and whether the template files exist
    if not path.isfile(dll):
        raise FileNotFoundError(f'SecStrAnnotator DLL "{dll}" not found.\nUse --dll option to set the path to SecStrAnnotator.dll.\n')
        # sys.stderr.write(f'Error: "{dll}" not found.\nUse --dll option to set the path to SecStrAnnotator.dll.\n')
        # return 1

    template_pdb = template.split(',')[0]
    template_struct_file = path.join(directory, template_pdb+'.cif')
    template_annot_file = path.join(directory, template_pdb+'-template.sses.json')
    if not onlyssa and not path.isfile(template_struct_file):
        raise FileNotFoundError(f'Template structure file "{template_struct_file}" not found\n')
        # sys.stderr.write(f'Error: Template structure file "{template_struct_file}" not found\n')
        # return 1
    if not onlyssa and not path.isfile(template_annot_file):
        raise FileNotFoundError(f'Template annotation file "{template_annot_file}" not found\n')
        # sys.stderr.write(f'Error: Template annotation file "{template_annot_file}" not found\n')
        # return 1

    secstrannotator_commands = ['dotnet', dll]

    # Prepare domain list
    domains = read_domains_json(queries_file)
    pdbs = sorted(domains)
    if not path.isdir(directory):
        raise NotADirectoryError(f'"{directory}" is not a directory\n')
    print(directory)
    n_domains = sum(len(domains[pdb]) for pdb in pdbs)
    if files_by_domain_name:
        found_pdbs = []
        not_found_pdbs = []
        found_domains = []
        not_found_domains = []
        for pdb, doms in domains.items():
            pdb_found = False
            for domain in doms:
                domain_name = domain[0]
                if path.isfile(path.join(directory, f'{domain_name}.cif')):
                    found_domains.append(domain)
                    pdb_found = True
                else:
                    not_found_domains.append(domain)
            if pdb_found:
                found_pdbs.append(pdb)
            else:
                not_found_pdbs.append(pdb)
        n_found_domains = len(found_domains)
    else:
        found_pdbs = [pdb for pdb in pdbs if path.isfile(path.join(directory, f'{pdb}.cif'))]
        not_found_pdbs = [pdb for pdb in pdbs if not path.isfile(path.join(directory, f'{pdb}.cif'))]
        n_found_domains = sum(len(domains[pdb]) for pdb in found_pdbs)
        not_found_domains = [domain for pdb in not_found_pdbs for domain in domains[pdb] ]
    
    not_found_domain_set = set(name for name, chain, rang in not_found_domains)

    print(f'Listed {len(pdbs)} PDBs ({n_domains} domains), found {len(found_pdbs)} PDBs ({n_found_domains} domains)')

    all_annotations = OrderedDict()
    failed = []

    def process_pdb(pdb):
        remove_file(path.join(directory, f'{pdb}.label2auth.tsv'))
        if by_pdb:
            process_domain(pdb, pdb)
        else:
            for domain_name, chain, ranges in domains[pdb]:
                process_domain(domain_name, pdb, chain, ranges)

    def process_domain(domain_name, pdb, chain=None, ranges=None):
        if domain_name in not_found_domain_set:
            return
        namebase = domain_name if files_by_domain_name else pdb
        query = namebase
        if chain is not None:
            query += ',' + chain 
        if ranges is not None:
            query += ',' + ranges
        thread_out = path.join(directory, f'stdout_thread_{threading.current_thread().name}.txt')
        thread_err = path.join(directory, f'stderr_thread_{threading.current_thread().name}.txt')
        with open(thread_out, 'a') as w_out:
            with open(thread_err, 'a') as w_err:
                w_out.write('\n' + '-'*70 + '\n' + domain_name + '\n')
                w_err.write('-'*70 + '\n' + domain_name + '\n')
                w_out.flush()
                w_err.flush()
                regular_arguments = [directory, template, query] if not onlyssa else [directory, query]
                arguments = secstrannotator_commands + regular_arguments + options
                exit_code = subprocess.call(arguments, stdout=w_out, stderr=w_err)
        if exit_code == 0:
            for ext in out_files_extensions:
                try_rename_file(path.join(directory, namebase + ext), path.join(directory, domain_name + ext))
            try:
                copy_file(path.join(directory, f'{namebase}-label2auth.tsv'), path.join(directory, f'{pdb}.label2auth.tsv'), append=True)
                remove_file(path.join(directory, f'{namebase}-label2auth.tsv'))
            except:
                pass
            if not onlyssa:
                annotation = try_read_json(path.join(directory, f'{domain_name}-annotated.sses.json')).get(pdb, {})
                annotation['domain'] = { 'name': domain_name, 'pdb': pdb, 'chain': chain, 'ranges': ranges }
                all_annotations[domain_name] = annotation
        else:
            failed.append(domain_name)

    def clear_outputs(thread):
        thread_out = path.join(directory, 'stdout_thread_' + thread.name + '.txt')
        thread_err = path.join(directory, 'stderr_thread_' + thread.name + '.txt')
        clear_file(thread_out)
        clear_file(thread_err)

    def merge_outputs(thread):
        thread_out = path.join(directory, 'stdout_thread_' + thread.name + '.txt')
        thread_err = path.join(directory, 'stderr_thread_' + thread.name + '.txt')
        copy_file(thread_out, output, append=True)
        copy_file(thread_err, output_err, append=True)
        remove_file(thread_out)
        remove_file(thread_err)

    clear_file(output)
    clear_file(output_err)

    progress_bar = ProgressBar(len(found_pdbs), title=f'Running SecStrAnnotator on {n_found_domains} domains', writer=sys.stderr)

    # Run SecStrAnnotator in parallel threads
    run_in_threads(process_pdb, found_pdbs, threads, progress_bar=progress_bar, initialize_thread_sync=clear_outputs, finalize_thread_sync=merge_outputs)
    
    all_annotations = { domain: annot for domain, annot in sorted(all_annotations.items()) }

    # Output collected data
    with open(all_annotations_file, 'w') as w:
        json.dump(all_annotations, w, indent=4)

    print('Failed to find ' + str(len(not_found_domains)) + ' domains:')
    print(', '.join(name for name, chain, ranges in not_found_domains))
    print('Failed to annotate ' + str(len(failed)) + ' domains:')
    print(', '.join(failed))
    print('Annotations in:  ' + all_annotations_file)
    print('Output in:       ' + output)
    print('Error output in: ' + output_err)


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)