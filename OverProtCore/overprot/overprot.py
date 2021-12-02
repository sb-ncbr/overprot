'''
This Python3 script does foo ...

Example usage:
    python3  overprot.py  FAMILY_ID  SAMPLE_SIZE  BASE_DIRECTORY 
'''

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from .libs import lib
from .libs import lib_sh
from .libs import lib_domains
from .libs import lib_sses
from .libs import lib_pymol
from .libs.lib import RedirectIO
from .libs.lib_overprot_config import OverProtConfig

from . import domains_from_pdbeapi
from . import select_random_domains
from . import remove_obsolete_pdbs
from . import run_mapsci
from . import mapsci_consensus_to_cif
from . import make_guide_tree
from . import acyclic_clustering
from . import draw_diagram
from . import cealign_all
from . import format_domains
from .libs.lib_dependencies import DEFAULT_CONFIG_FILE, STRUCTURE_CUTTER_DLL


def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', help='Family identifier for PDBe API (e.g. CATH code)', type=str)
    parser.add_argument('sample_size', help='Number of domains to process (integer or "all")', type=str)
    parser.add_argument('directory', help='Directory to save everything in', type=Path)
    parser.add_argument('--config', help=f'Configuration file (default: {DEFAULT_CONFIG_FILE})', type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--domains', help='File with the list of input domains (do not download domain list)', type=Path)
    parser.add_argument('--structure_source', help='Prepend a structure source to download.structure_sources parameter from the config file.', type=str)
    parser.add_argument('--out', help='File for stdout.', type=Path)
    parser.add_argument('--err', help='File for stderr.', type=Path)
    args = parser.parse_args()
    if args.sample_size != 'all':
        try:
            args.sample_size = int(args.sample_size)
        except ValueError:
            parser.error(f"argument sample_size: invalid value: '{args.sample_size}', must be int or 'all'")
    return vars(args)


def main(family: str, sample_size: int|str|None, directory: Path, config: Optional[Path] = DEFAULT_CONFIG_FILE, 
         domains: Optional[Path] = None, structure_source: Optional[str] = None, 
         out: Optional[Path] = None, err: Optional[Path] = None) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    
    if config is None:
        config = DEFAULT_CONFIG_FILE
    
    datadir = directory  #TODO rename datadir to directory
    
    conf = OverProtConfig(config, allow_extra=False, allow_missing=False)
    if structure_source is not None and structure_source != '':
        conf.download.structure_sources.insert(0, structure_source)
    results = conf.files.results_dir

    print('Configuration:', config)
    print('Output directory:', datadir)

    # Download the list of domains in the family
    print('::: GET DOMAIN LIST :::')
    datadir.mkdir(parents=True, exist_ok=True)
    with open(datadir/'family_info.txt', 'w') as w:
        print('family_id:', family, file=w)
    if domains is None:
        with RedirectIO(stdout=datadir/'family-orig.json'):
            domains_from_pdbeapi.main(family, join_domains_in_chain=True)
    else:
        lib_domains.save_domain_list(lib_domains.load_domain_list_by_pdb(domains), datadir/'family-orig.json')
    domains_by_pdb = lib_domains.load_domain_list_by_pdb(datadir/'family-orig.json')
    n_pdbs = len(domains_by_pdb)
    n_domains = sum(len(doms) for doms in domains_by_pdb.values())
    with open(datadir/'family_info.txt', 'a') as w:
        print('n_pdbs:', n_pdbs, file=w)
        print('n_domains:', n_domains, file=w)

    lib_sh.cp(datadir/'family-orig.json', datadir/'family.json')
    # python3  domains_with_observed_residues.py  --min_residues 4  $DIR/family-orig.json  >  $DIR/family.json  # TODO solve without an API call for each PDB
    
    # # Create a list of domains based on DALI results
    # echo '::: PREPARE DOMAIN LIST :::'
    # mkdir  -p  $DIR
    # python3  dali_hits_to_domains.py  --max_loop 50  $DIR/DALI.txt  > $DIR/family-auth.json
    # python3  convert_domain_numbering.py  --ignore_obsolete  auth2label  $DIR/family-auth.json  > $DIR/family.json
    # TODO decide about this branch based on config (not comment-out) (or just keep in dev)

    # Select random sample from the family
    print('\n::: SELECT SAMPLE :::')
    with RedirectIO(stdout=datadir/'sample.json'):
        select_random_domains.main(datadir/'family.json', size=sample_size, or_all=conf.sample_selection.or_all, unique_pdb=conf.sample_selection.unique_pdb)
    if conf.overprot.annotate_whole_family:
        with RedirectIO(stdout=datadir/'sample-whole_family.json'):
            select_random_domains.main(datadir/'family.json', size='all', unique_pdb=False)
        
    sample_domains = lib_domains.load_domain_list(datadir/'sample.json')
    n_sample = len(sample_domains)
    with open(datadir/'family_info.txt', 'a') as w:
        print('n_sample:', n_sample, file=w)
    
    # Download structures in CIF, cut the domains and save them as CIF and PDB
    print('\n::: DOWNLOAD :::')
    sample_for_annotation = datadir/'sample-whole_family.json' if conf.overprot.annotate_whole_family else datadir/'sample.json'
    lib.run_dotnet(STRUCTURE_CUTTER_DLL, sample_for_annotation, '--sources', ' '.join(conf.download.structure_sources), 
                   '--cif_outdir', datadir/'cif', '--pdb_outdir', datadir/'pdb', 
                   '--failures', datadir/'StructureCutter-failures.txt', 
                   timing=True) 
    
    # Check if some failed-to-download structures are obsolete or what; remove them from the sample if yes
    some_still_missing = remove_obsolete_pdbs.main(datadir/'sample.json', datadir/'StructureCutter-failures.txt', 
                                                   datadir/'sample-nonobsolete.json', datadir/'StructureCutter-failures-nonobsolete.txt')
    if some_still_missing != 0:
        raise Exception(f'Failed to download some structure and they are not obsolete. See {datadir/"StructureCutter-failures-nonobsolete.txt"}')
    else:
        lib_sh.mv(datadir/'sample.json', datadir/'sample-original.json')
        lib_sh.mv(datadir/'sample-nonobsolete.json', datadir/'sample.json')
    sample_domains = lib_domains.load_domain_list(datadir/'sample.json')
    n_sample = len(sample_domains)
    with open(datadir/'family_info.txt', 'a') as w:
        print('n_sample_without_obsoleted:', n_sample, file=w)

    # Convert PDB and domain lists into various formats
    format_domains.main(datadir/'family.json', datadir/'sample.json', out_dir=datadir/'lists')
    lib_sh.cp(datadir/'family_info.txt', datadir/'lists'/'family_info.txt')
    if n_sample == 0:
        (datadir/'EMPTY_FAMILY').write_text('')
        return 1

    # Perform multiple structure alignment by MAPSCI
    print('\n::: MAPSCI :::')
    run_mapsci.main(datadir/'sample.json', datadir/'pdb', datadir/'mapsci', init=conf.mapsci.init, n_max=conf.mapsci.n_max)
    mapsci_consensus_to_cif.main(datadir/'mapsci'/'consensus.pdb', datadir/'mapsci'/'consensus.cif')

    # Align structures to the consensus backbone by PyMOL's CEalign
    print('\n::: CEALIGN :::')
    cealign_all.main(datadir/'mapsci'/'consensus.cif', sample_for_annotation, datadir/'cif', datadir/'cif_cealign', progress_bar=True)
    lib_sh.cp(datadir/'mapsci'/'consensus.cif', datadir/'cif_cealign'/'consensus.cif')

    if conf.files.clean_pdb_cif:
        lib_sh.rm(datadir/'pdb', recursive=True)
        lib_sh.rm(datadir/'cif', recursive=True)

    # Calculate SSE positions and cluster SSEs
    # Will add manual labels into the consensus annotation, if $DIR/manual-*.sses.json and $DIR/manual-*.cif exist
    print('\n::: CLUSTERING :::')
    lib_sses.compute_ssa(sample_for_annotation, datadir/'cif_cealign', skip_if_exists = not conf.overprot.force_ssa, progress_bar=True)
    lib_sh.cp(datadir/'sample.json', datadir/'cif_cealign'/'sample.json')
    with RedirectIO(tee_stdout=datadir/'making_guide_tree.log'):
        make_guide_tree.main(datadir/'cif_cealign', show_tree=False, progress_bar=True)
    with RedirectIO(tee_stdout=datadir/'clustering.log'):
        acyclic_clustering.main(datadir/'cif_cealign', force_ssa=conf.overprot.force_ssa, secstrannotator_rematching=conf.overprot.secstrannotator_rematching, min_occurrence=0, fallback=60)
    if conf.overprot.annotate_whole_family:
        lib_sses.annotate_all_with_SecStrAnnotator([dom for doms in domains_by_pdb.values() for dom in doms], datadir/'cif_cealign', extra_options='--fallback 60', outdirectory=datadir/'annotated_sses')

    # Tidy up
    (datadir/results).mkdir(parents=True, exist_ok=True)
    for filename in str.split('lengths.tsv cluster_precedence_matrix.tsv statistics.tsv occurrence_correlation.tsv consensus.cif consensus.sses.json'):
        lib_sh.cp(datadir/'cif_cealign'/filename, datadir/results/filename)

    # Draw diagrams
    print('\n::: DIAGRAM :::')
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram.svg', json_output=datadir/results/'diagram.json')
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_min20.svg', occurrence_threshold=0.2)
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_conn.svg', connectivity=True)
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_conn_min20.svg', connectivity=True, occurrence_threshold=0.2)
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_conn_symcdf.svg', connectivity=True, shape='symcdf')
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_conn_symcdf_min20.svg', connectivity=True, shape='symcdf', occurrence_threshold=0.2)
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_conn_arrow_min20.svg', connectivity=True, shape='arrow', occurrence_threshold=0.2)
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_conn_heatmap.svg', connectivity=True, heatmap=True)
    draw_diagram.main(datadir/results, dag=True, output=datadir/results/'diagram_conn_heatmap_min20.svg', connectivity=True, heatmap=True, occurrence_threshold=0.2)

    # Visualize in PyMOL and save as a session + PNG
    print('\n::: VISUALIZE :::')
    lib_pymol.create_consensus_session(datadir/results/'consensus.cif', datadir/results/'consensus.sses.json', datadir/results/'consensus.pse', 
                                       coloring=conf.visualization.coloring, out_image_file=datadir/results/'consensus.png', image_size=(1000,))
    if conf.visualization.create_multi_session:
        lib_pymol.create_multi_session(datadir/'cif_cealign', datadir/results/'consensus.cif', datadir/results/'consensus.sses.json', 
                                       datadir/results/'clustered.pse', coloring=conf.visualization.coloring, progress_bar=True)

    if conf.files.clean_aligned_cif:
        lib_sh.rm(*(datadir/'cif_cealign').glob('*.cif'))

    print('\n::: COMPLETED :::')
    # TODO write docs for each module/script, update README.txt
    return None


def _main():
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)


if __name__ == '__main__':
    _main()
