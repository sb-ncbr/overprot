'''
This Python3 script does foo ...

Example usage:
    python3  overprot.py  FAMILY_ID  SAMPLE_SIZE  BASE_DIRECTORY 
'''

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .libs import lib
from .libs import lib_domains
from .libs import lib_sses
from .libs import lib_pymol
from .libs.lib import RedirectIO, FilePath
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
    parser.add_argument('directory', help='Directory to save everything in', type=str)
    parser.add_argument('--config', help=f'Configuration file (default: {DEFAULT_CONFIG_FILE})', type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--domains', help='File with the list of input domains (do not download domain list)', type=str)
    parser.add_argument('--structure_source', help='Prepend a structure source to download.structure_sources parameter from the config file.', type=str)
    parser.add_argument('--out', help='File for stdout.', type=str)
    parser.add_argument('--err', help='File for stderr.', type=str)
    args = parser.parse_args()
    if args.sample_size != 'all':
        try:
            args.sample_size = int(args.sample_size)
        except ValueError:
            parser.error(f"argument sample_size: invalid value: '{args.sample_size}', must be int or 'all'")
    return vars(args)


def main(family: str, sample_size: Union[int, str, None], directory: Union[FilePath, str], config: Optional[str] = DEFAULT_CONFIG_FILE, 
         domains: Union[FilePath, str, None] = None, structure_source: Optional[str] = None, 
         out: Optional[str] = None, err: Optional[str] = None) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    
    if config is None:
        config = DEFAULT_CONFIG_FILE
    
    datadir = FilePath(directory)
    
    conf = OverProtConfig(config, allow_extra=False, allow_missing=False)
    if structure_source is not None and structure_source != '':
        conf.download.structure_sources.insert(0, structure_source)
    results = conf.files.results_dir

    print('Configuration:', config)
    print('Output directory:', datadir)

    # Download the list of domains in the family
    print('::: GET DOMAIN LIST :::')
    datadir._mkdir(exist_ok=True)
    with datadir._sub('family_info.txt')._open('w') as w:
        print('family_id:', family, file=w)
    if domains is None:
        with RedirectIO(stdout=datadir._sub('family-orig.json')):
            domains_from_pdbeapi.main(family, join_domains_in_chain=True)
    else:
        lib_domains.save_domain_list(lib_domains.load_domain_list_by_pdb(domains), datadir._sub('family-orig.json'))
    domains_by_pdb = lib_domains.load_domain_list_by_pdb(datadir._sub('family-orig.json'))
    n_pdbs = len(domains_by_pdb)
    n_domains = sum(len(doms) for doms in domains_by_pdb.values())
    with datadir._sub('family_info.txt')._open('a') as w:
        print('n_pdbs:', n_pdbs, file=w)
        print('n_domains:', n_domains, file=w)

    datadir._sub('family-orig.json').cp(datadir._sub('family.json'))
    # python3  domains_with_observed_residues.py  --min_residues 4  $DIR/family-orig.json  >  $DIR/family.json  # TODO solve without an API call for each PDB
    
    # # Create a list of domains based on DALI results
    # echo '::: PREPARE DOMAIN LIST :::'
    # mkdir  -p  $DIR
    # python3  dali_hits_to_domains.py  --max_loop 50  $DIR/DALI.txt  > $DIR/family-auth.json
    # python3  convert_domain_numbering.py  --ignore_obsolete  auth2label  $DIR/family-auth.json  > $DIR/family.json
    # TODO decide about this branch based on config (not comment-out) (or just keep in dev)

    # Select random sample from the family
    print('\n::: SELECT SAMPLE :::')
    with RedirectIO(stdout=datadir._sub('sample.json')):
        select_random_domains.main(datadir._sub('family.json'), size=sample_size, or_all=conf.sample_selection.or_all, unique_pdb=conf.sample_selection.unique_pdb)
    if conf.overprot.annotate_whole_family:
        with RedirectIO(stdout=datadir._sub('sample-whole_family.json')):
            select_random_domains.main(datadir._sub('family.json'), size='all', unique_pdb=False)
        
    sample_domains = lib_domains.load_domain_list(datadir._sub('sample.json'))
    n_sample = len(sample_domains)
    with datadir._sub('family_info.txt')._open('a') as w:
        print('n_sample:', n_sample, file=w)
    
    # Download structures in CIF, cut the domains and save them as CIF and PDB
    print('\n::: DOWNLOAD :::')
    sample_for_annotation = datadir._sub('sample-whole_family.json' if conf.overprot.annotate_whole_family else 'sample.json')
    lib.run_dotnet(STRUCTURE_CUTTER_DLL, sample_for_annotation, '--sources', ' '.join(conf.download.structure_sources), 
                   '--cif_outdir', datadir._sub('cif'), '--pdb_outdir', datadir._sub('pdb'), 
                   '--failures', datadir._sub('StructureCutter-failures.txt'), 
                   timing=True) 
    
    # Check if some failed-to-download structures are obsolete or what; remove them from the sample if yes
    some_still_missing = remove_obsolete_pdbs.main(datadir._sub('sample.json'), datadir._sub('StructureCutter-failures.txt'), 
                                                   datadir._sub('sample-nonobsolete.json'), datadir._sub('StructureCutter-failures-nonobsolete.txt'))
    if some_still_missing != 0:
        raise Exception(f'Failed to download some structure and they are not obsolete. See {datadir._sub("StructureCutter-failures-nonobsolete.txt")}')
    else:
        datadir._sub('sample.json').mv(datadir._sub('sample-original.json'))
        datadir._sub('sample-nonobsolete.json').mv(datadir._sub('sample.json'))
    sample_domains = lib_domains.load_domain_list(datadir._sub('sample.json'))
    n_sample = len(sample_domains)
    with datadir._sub('family_info.txt')._open('a') as w:
        print('n_sample_without_obsoleted:', n_sample, file=w)

    # Convert PDB and domain lists into various formats
    format_domains.main(Path(datadir._sub('family.json')), Path(datadir._sub('sample.json')), out_dir=Path(datadir._sub('lists')))
    datadir._sub('family_info.txt').cp(datadir._sub('lists', 'family_info.txt'))
    if n_sample == 0:
        datadir._sub('EMPTY_FAMILY').clear()
        return 1

    # Perform multiple structure alignment by MAPSCI
    print('\n::: MAPSCI :::')
    run_mapsci.main(datadir._sub('sample.json'), datadir._sub('pdb'), datadir._sub('mapsci'), init=conf.mapsci.init, n_max=conf.mapsci.n_max)
    mapsci_consensus_to_cif.main(datadir._sub('mapsci', 'consensus.pdb'), datadir._sub('mapsci', 'consensus.cif'))

    # Align structures to the consensus backbone by PyMOL's CEalign
    print('\n::: CEALIGN :::')
    cealign_all.main(datadir._sub('mapsci', 'consensus.cif'), sample_for_annotation, datadir._sub('cif'), datadir._sub('cif_cealign'), progress_bar=True)
    datadir._sub('mapsci', 'consensus.cif').cp(datadir._sub('cif_cealign', 'consensus.cif'))

    if conf.files.clean_pdb_cif:
        datadir._sub('pdb').rm(recursive=True)
        datadir._sub('cif').rm(recursive=True)

    # Calculate SSE positions and cluster SSEs
    # Will add manual labels into the consensus annotation, if $DIR/manual-*.sses.json and $DIR/manual-*.cif exist
    print('\n::: CLUSTERING :::')
    lib_sses.compute_ssa(sample_for_annotation, datadir._sub('cif_cealign'), skip_if_exists = not conf.overprot.force_ssa, progress_bar=True)
    datadir._sub('sample.json').cp(datadir._sub('cif_cealign', 'sample.json'))
    with RedirectIO(tee_stdout=datadir._sub('making_guide_tree.log')):
        make_guide_tree.main(datadir._sub('cif_cealign'), show_tree=False, progress_bar=True)
    with RedirectIO(tee_stdout=datadir._sub('clustering.log')):
        acyclic_clustering.main(datadir._sub('cif_cealign'), force_ssa=conf.overprot.force_ssa, secstrannotator_rematching=conf.overprot.secstrannotator_rematching, min_occurrence=0, fallback=60)
    if conf.overprot.annotate_whole_family:
        lib_sses.annotate_all_with_SecStrAnnotator([dom for doms in domains_by_pdb.values() for dom in doms], datadir._sub('cif_cealign'), extra_options='--fallback 60', outdirectory=datadir._sub('annotated_sses'))

    # Tidy up
    datadir._mkdir(results, exist_ok=True)
    for filename in str.split('lengths.tsv cluster_precedence_matrix.tsv statistics.tsv occurrence_correlation.tsv consensus.cif consensus.sses.json'):
        datadir._sub('cif_cealign', filename).cp(datadir._sub(results, filename))

    # Draw diagrams
    print('\n::: DIAGRAM :::')
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram.svg'), json_output=datadir._sub(results, 'diagram.json'))
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_min20.svg'), occurrence_threshold=0.2)
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_conn.svg'), connectivity=True)
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_conn_min20.svg'), connectivity=True, occurrence_threshold=0.2)
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_conn_symcdf.svg'), connectivity=True, shape='symcdf')
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_conn_symcdf_min20.svg'), connectivity=True, shape='symcdf', occurrence_threshold=0.2)
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_conn_arrow_min20.svg'), connectivity=True, shape='arrow', occurrence_threshold=0.2)
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_conn_heatmap.svg'), connectivity=True, heatmap=True)
    draw_diagram.main(datadir._sub(results), dag=True, output=datadir._sub(results, 'diagram_conn_heatmap_min20.svg'), connectivity=True, heatmap=True, occurrence_threshold=0.2)

    # Visualize in PyMOL and save as a session + PNG
    print('\n::: VISUALIZE :::')
    lib_pymol.create_consensus_session(datadir._sub(results, 'consensus.cif'), datadir._sub(results, 'consensus.sses.json'), datadir._sub(results, 'consensus.pse'), 
                                       coloring=conf.visualization.coloring, out_image_file=datadir._sub(results, 'consensus.png'), image_size=(1000,))
    if conf.visualization.create_multi_session:
        lib_pymol.create_multi_session(datadir._sub('cif_cealign'), datadir._sub(results, 'consensus.cif'), datadir._sub(results, 'consensus.sses.json'), 
                                       datadir._sub(results, 'clustered.pse'), coloring=conf.visualization.coloring, progress_bar=True)

    if conf.files.clean_aligned_cif:
        for file in datadir._sub('cif_cealign').ls(only_files=True):
            if str(file).endswith('.cif'):
                file.rm()

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
