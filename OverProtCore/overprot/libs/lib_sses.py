'''Library of functions related to SSEs (secondary structure elements) and running SecStrAnnotator'''

from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import json
import re
import itertools
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from enum import IntEnum

from . import lib
from . import lib_sh
from . import lib_domains
from .lib_domains import Domain
from .lib_dependencies import PYTHON, SECSTRANNOTATOR_DLL, SECSTRANNOTATOR_BATCH_PY

# Field names in JSON annotation format
LABEL = 'label'
CHAIN = 'chain_id'
START = 'start'
END = 'end' # TODO move these into a separate module, which can then be imported as 'from sse_field_names import *'


class SseType(IntEnum):
    HELIX = 0
    STRAND = 1

class LadderType(IntEnum):
    PARALLEL = 1
    ANTIPARALLEL = -1


def get_out_err_files(directory: Path, append: bool = True):
    stdout_file = directory/'stdout.txt'
    stderr_file = directory/'stderr.txt'
    if not append:
        stdout_file.write_text('')
        stderr_file.write_text('')
    return stdout_file, stderr_file

def compute_ssa(domains: List[Domain]|Path, directory: Path, skip_if_exists=False, append_outputs=False, 
                create_template_files=False, progress_bar=False):
    lib.log(f'Compute SSA: {directory}')
    if isinstance(domains, Path):
        domains = lib_domains.load_domain_list(domains)
    sse_files = [directory/f'{domain.name}.sses.json' for domain in domains]
    detected_sse_files = [directory/f'{domain.name}-detected.sses.json' for domain in domains]
    template_sse_files = [directory/f'{domain.name}-template.sses.json' for domain in domains]

    if skip_if_exists and all(f.is_file() for f in sse_files):
        lib.log(f'Skipping SSA computation (all files already exist)')
    else:
        stdout_file, stderr_file = get_out_err_files(directory, append=append_outputs)
        with lib.Timing('SSA'):
            with lib.ProgressBar(len(domains), title='Computing SSA', mute = not progress_bar) as bar:
                for domain, detected_file, sse_file in zip(domains, detected_sse_files, sse_files):
                    append_to_file(stdout_file, f'>>> {domain.name},{domain.chain}\n')
                    append_to_file(stderr_file, f'>>> {domain.name},{domain.chain}\n')
                    lib.run_dotnet(SECSTRANNOTATOR_DLL, '--onlyssa', '--verbose', directory, f'{domain.name},{domain.chain}', 
                        stdout=stdout_file, stderr=stderr_file, appendout=True, appenderr=True)
                    lib_sh.mv(detected_file, sse_file)
                    bar.step()

    failed_domains = [domain.name for domain, sse_file in zip(domains, sse_files) if not sse_file.is_file()]
    if len(failed_domains) > 0:
        failed_domains_str = lib.str_join(', ', failed_domains, three_dots_after=5)
        raise ChildProcessError(f'Secondary structure assignment failed for {len(failed_domains)} domains ({failed_domains_str})')
    if create_template_files:
        for sse_file, template_sse_file in zip(sse_files, template_sse_files):
            shutil.copy(sse_file, template_sse_file)

def compute_distance_matrices(samples, directory: Path, append_outputs: bool = True):
    stdout_file, stderr_file = get_out_err_files(directory, append=append_outputs)
    n = len(samples)
    with lib.ProgressBar(n*(n-1)//2, title='Computing distance matrices') as bar:
        for i, j in itertools.combinations_with_replacement(range(len(samples)), 2):
            pi, di, ci, ri = samples[i]
            pj, dj, cj, rj = samples[j]
            lib.run_dotnet(SECSTRANNOTATOR_DLL, '--ssa', 'file', '--matching', 'none', directory, f'{di},{ci},{ri}' , f'{dj},{cj},{rj}', stdout=stdout_file, stderr=stderr_file, appendout=True, appenderr=True)
            lib_sh.mv(directory/'metric_matrix.tsv', directory/f'matrix-{di}-{dj}.tsv')
            lib_sh.rm(directory/f'alignment-{di}-{dj}.json', ignore_errors=True)
            lib_sh.rm(directory/f'{dj}-detected.sses.json', ignore_errors=True)
            bar.step()
        lib_sh.rm(directory / 'template-smooth.pdb', ignore_errors=True)

def annotate_all_with_SecStrAnnotator(domains: List[Domain], directory: Path, append_outputs=True, extra_options='', outdirectory: Optional[Path] = None):
    samples_by_pdb = { domain.name: [(domain.name, domain.chain, domain.ranges)] for domain in domains }
    lib.dump_json(samples_by_pdb, directory/'samples_by_pdb.json', minify=True)
    shutil.copy(directory/'consensus.sses.json', directory/'consensus-template.sses.json')
    options = '--ssa file  --align none  --metrictype 3 ' + extra_options
    print('Running SecStrAnnotator:')
    lib.run_command(PYTHON, SECSTRANNOTATOR_BATCH_PY, 
        '--dll', SECSTRANNOTATOR_DLL, 
        '--threads', '8', 
        '--options', f'{options} ', 
        directory, 
        'consensus',
        directory/'samples_by_pdb.json',
        timing=True)
    print()
    if outdirectory is not None:
        outdirectory.mkdir(parents=True, exist_ok=True)
        for domain in domains:
            filename = f'{domain.name}-annotated.sses.json'
            lib_sh.mv(directory/filename, outdirectory/filename)
            lib_sh.rm(directory/f'{domain.name}-detected.sses.json')
            lib_sh.rm(directory/f'{domain.name}-aligned.cif')


def map_manual_template_to_consensus(directory: Path):
    manuals = sorted(directory.parent.glob('manual*.sses.json'))
    if len(manuals) == 0:
        pass
    elif len(manuals) > 1:
        lib.log(f'WARNING: More than one manual annotation file found, not using any: {manuals}')
    else:
        manual_file = lib.single(manuals)
        chain = get_chain_from_annotation(manual_file)
        domain_match = re.search('^(manual.*)\.sses\.json$', manual_file.name)
        assert domain_match is not None
        domain = domain_match.group(1)
        lib_sh.cp(directory.parent/f'{domain}.sses.json', directory/ f'{domain}.sses.json')
        lib_sh.cp(directory.parent/f'{domain}.cif', directory/f'{domain}.cif')
        lib_sh.cp(directory/'consensus.sses.json', directory/'consensus-template.sses.json')
        stdout_file, stderr_file = get_out_err_files(directory, append=True)
        lib.run_dotnet(SECSTRANNOTATOR_DLL, '--ssa', 'file', '--metrictype', '3', '--verbose', directory, 'consensus', f'{domain},{chain}',
            stdout=stdout_file, stderr=stderr_file, appendout=True, appenderr=True)
        label_mapping = {}
        with open(directory/f'matching-consensus-{domain}.tsv') as r:
            r.readline()  # header
            for line in r:
                automatic, manual = line.strip().split('\t')
                label_mapping[automatic] = manual
        with open(directory/'consensus.sses.json') as r:
            js = json.load(r)
        mapped = 0
        for sse in js['consensus']['secondary_structure_elements']:
            automatic = sse['label']
            if automatic in label_mapping:
                manual_label = label_mapping[automatic]
                lib.insert_after(sse, 'label', [('manual_label', manual_label)])
                mapped += 1
        lib.dump_json(js, directory/'consensus.sses.json')
        print(f'Mapped {mapped} manual labels out of {len(label_mapping)}')

def get_chain_from_annotation(annotation_file: Path) -> str:
    with open(annotation_file) as r:
        js = json.load(r)
    annot = next(iter(js.values()))
    chains = { sse['chain_id'] for sse in annot['secondary_structure_elements'] }
    try:
        chain = lib.single(chains)
        return chain
    except ValueError as e:
        raise Exception(f'Annotation file {annotation_file} does not contain exactly one chain ({", ".join(chains)}; {str(e)})')

def sort_samples_by_pdb(samples):
    result = defaultdict(lambda: [])
    for pdb, domain, chain, ranges in samples:
        result[pdb].append((domain, chain, ranges))
    return result

def length(sse):
    '''Return the length (i.e. number of residues) of a SSE (represented as JSON object).'''
    return sse['end'] - sse['start'] + 1 if sse is not None else 0

def two_class_type(sse) -> str:
    '''Convert SSE type from multi-class to two-class (i.e. H or E).'''
    if sse['type'] in 'GHIh':
        return 'H'
    elif sse['type'] in 'EBe':
        return 'E'
    else:
        raise Exception('Unknown secondary structure type: ' + sse['type'])

def two_class_type_int(sse) -> SseType:
    return SseType.STRAND if two_class_type(sse) == 'E' else SseType.HELIX

def long_enough(sse, thresholds):
    return thresholds is None or length(sse) >= thresholds[two_class_type(sse)]

def hash_color(i: int) -> str:
    K = 4 # appr. number of colors that span 1 rainbow
    ALPHA = 1.0 / K + np.log(2.0) / K**2
    num_color = int(i * ALPHA * 1000) % 1000
    return 's' + str(num_color).zfill(3)

def pymol_spectrum_to_rgb(color: str) -> Tuple[float, float, float]:
    '''Convert color representation from PyMOL spectrum to RGB, e.g. 's500' -> (0.0, 1.0, 0.0) (green).'''
    if color.startswith('s') and len(color) == 4 and color[1:].isdigit():
        i = int(color[1:])
    else:
        raise ValueError("PyMOL spectrum color must be in format 'sXYZ' (e.g. 's523'), not '" + color + "'")
    sector = i * 6 // 1000  # The 6 sectors are: mb, bc, cg, gy, yr, rm
    x = i * 6 % 1000 / 1000  # Position within sector [0.0, 1.0)
    if sector == 0:  # magenta-blue
        return (1.0 - x, 0.0, 1.0)
    elif sector == 1:  # blue-cyan
        return (0.0, x, 1.0)
    elif sector == 2:  # cyan-green
        return (0.0, 1.0, 1.0 - x)
    elif sector == 3:  # green-yellow
        return (x, 1.0, 0.0)
    elif sector == 4:  # yellow-red
        return (1.0, 1.0 - x, 0.0)
    elif sector == 5:  # red-magenta
        return (1.0, 0.0, x)
    else:
        raise AssertionError('sector must be 0, 1, 2, 3, 4, or 5')

def pymol_spectrum_to_hex(color: str) -> str:
    '''Convert color representation from PyMOL spectrum to hexadecimal representation, e.g. 's500' -> #00FF00 (green).'''
    r, g, b = pymol_spectrum_to_rgb(color)
    R = int(255*r)
    G = int(255*g)
    B = int(255*b)
    return '#' + hex2(R) + hex2(G) + hex2(B)

def hex2(number: int):
    '''Get two-digit hexadecimal representation of integer from [0, 255], e.g. 10 -> '0A'.'''
    return hex(number)[2:].zfill(2).upper()
    

def spectrum_color(x: float) -> str:
    '''Assign color to number x (0.0 <= x <= 1.0)'''
    FROM, TO = 100, 900
    num_color = int(FROM + (TO - FROM) * x)  # n elements span almost the whole rainbow (from 100-900, to distinguish start from end)
    # num_color = (num_color + 500) % 1000  # Retarded inversion  # Debug
    return 's' + str(num_color).zfill(3)

def spectrum_color_i(i: int, n: int) -> str:
    return spectrum_color(i / (n-1))

def spectrum_colors(names: List[str]) -> Dict[str, str]:
    n = len(names)
    color_dict = { name: spectrum_color_i(i, n) for i, name in enumerate(names) }
    return color_dict

def spectrum_colors_weighted(weights: List[float]) -> List[str]:
    total = sum(weights)
    weights = [w / total for w in weights]
    cum = 0.0
    colors = []
    for weight in weights:
        colors.append(spectrum_color(cum + 0.5*weight))
        cum += weight
    return colors

def append_to_file(filename: Path, string):
    with open(filename, 'a') as w:
        w.write(string)

def filter_annotation(input_file: Path, indices: List[int], output_file: Path, recolor: bool = True):
    with open(input_file) as r:
        annotations = json.load(r)
    output = {}
    for name, annotation in annotations.items():
        sses = annotation['secondary_structure_elements']
        connectivity = annotation['beta_connectivity']
        sses = [ sses[i] for i in indices ]
        labels = set( sse['label'] for sse in sses )
        connectivity = [ conn for conn in connectivity if conn[0] in labels and conn[1] in labels ]
        if recolor:
            n = len(sses)
            for i, sse in enumerate(sses):
                sse['color'] = spectrum_color_i(i, n)
        output[name] = { 'secondary_structure_elements': sses, 'beta_connectivity': connectivity }
    lib.dump_json(output, output_file)