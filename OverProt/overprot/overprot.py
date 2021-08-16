'''
This Python3 script does foo ...

Example usage:
    python3  overprot.py  FAMILY_ID  SAMPLE_SIZE  BASE_DIRECTORY 
'''
# TODO Continue mypying (str -> FilePath)

# TODO add description and example usage in docstring

# # Prerequisities
# # python >= 3.6
# # dotnet: https://dotnet.microsoft.com/download 
# # Install necessary dotnet packages by building StructureCutter:
# # cd StructureCutter; dotnet build
# # sudo apt install python3-pip
# # pip3 install numpy svgwrite [matplotlib ete3]
# # sudo apt install pymol

import argparse
from typing import Dict, Any, Optional, Union, List, Literal

from .libs import lib
from .libs import lib_domains
from .libs import lib_sses
from .libs import lib_pymol
from .libs.lib import RedirectIO, Config, ConfigSection, FilePath

from . import domains_from_pdbeapi
from . import select_random_domains
from . import simplify_domain_list
from . import remove_obsolete_pdbs
from . import run_mapsci
from . import mapsci_consensus_to_cif
from . import make_guide_tree
from . import acyclic_clustering
from . import draw_diagram
from . import cealign_all
from . import format_domains


#  CONSTANTS  ################################################################################

DEFAULT_CONFIG_FILE = str(FilePath(__file__).parent().parent().sub('overprot-config.ini'))

#  FUNCTIONS  ################################################################################

class OverProtConfig(Config):
    class DownloadCS(ConfigSection):
        structure_cutter_path: str
        structure_sources: List[str]
    class SampleSelectionCS(ConfigSection):
        unique_pdb: bool
        or_all: bool
    class MapsciCS(ConfigSection):
        mapsci_path: str
        init: Literal['median', 'center']
        n_max: int
    class OverProtCS(ConfigSection):
        force_ssa: bool
        secstrannotator_rematching: bool
        secstrannotator_path: str
    class FilesCS(ConfigSection):
        results_dir: str
        clean_pdb_cif: bool
    class VisualizationCS(ConfigSection):
        coloring: Literal['color', 'rainbow']
        create_multi_session: bool

    download: DownloadCS
    sample_selection: SampleSelectionCS
    mapsci: MapsciCS
    sec_str_consensus: OverProtCS
    files: FilesCS
    visualization: VisualizationCS

#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', help='Family identifier for PDBe API (e.g. CATH code)', type=str)
    parser.add_argument('sample_size', help='Number of domains to process (integer or "all")', type=str)
    parser.add_argument('directory', help='Directory to save everything in', type=str)
    parser.add_argument('--config', help=f'Configuration file (default: {DEFAULT_CONFIG_FILE})', type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--domains', help='File with the list of input domains (do not download domain list)', type=str)
    args = parser.parse_args()
    return vars(args)


def main(family: str, sample_size: Union[int, str, None], directory: Union[FilePath, str], config: Optional[str] = DEFAULT_CONFIG_FILE, domains: Union[FilePath, str, None] = None) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    
    if config is None:
        config = DEFAULT_CONFIG_FILE
    
    datadir = FilePath(directory)
    
    conf = OverProtConfig(config, allow_extra=False, allow_missing=False)
    conf.download.structure_cutter_path = str(FilePath(config).parent().sub(conf.download.structure_cutter_path))
    conf.mapsci.mapsci_path = str(FilePath(config).parent().sub(conf.mapsci.mapsci_path))
    conf.sec_str_consensus.secstrannotator_path = str(FilePath(config).parent().sub(conf.sec_str_consensus.secstrannotator_path))
    results = conf.files.results_dir

    print('Configuration:', config)
    print('Output directory:', datadir)

    # Download the list of domains in the family
    print('::: GET DOMAIN LIST :::')
    datadir.mkdir(exist_ok=True)
    with datadir.sub('family_info.txt').open('w') as w:
        print('family_id:', family, file=w)
    if domains is None:
        with RedirectIO(stdout=datadir.sub('family-orig.json')):
            domains_from_pdbeapi.main(family, join_domains_in_chain=True)
    else:
        lib_domains.save_domain_list(lib_domains.load_domain_list_by_pdb(domains), datadir.sub('family-orig.json'))
    domains_by_pdb = lib_domains.load_domain_list_by_pdb(datadir.sub('family-orig.json'))
    n_pdbs = len(domains_by_pdb)
    n_domains = sum(len(doms) for doms in domains_by_pdb.values())
    with datadir.sub('family_info.txt').open('a') as w:
        print('n_pdbs:', n_pdbs, file=w)
        print('n_domains:', n_domains, file=w)

    datadir.sub('family-orig.json').cp(datadir.sub('family.json'))
    # python3  domains_with_observed_residues.py  --min_residues 4  $DIR/family-orig.json  >  $DIR/family.json  # TODO solve without an API call for each PDB
    
    # # Create a list of domains based on DALI results
    # echo '::: PREPARE DOMAIN LIST :::'
    # mkdir  -p  $DIR
    # python3  dali_hits_to_domains.py  --max_loop 50  $DIR/DALI.txt  > $DIR/family-auth.json
    # python3  convert_domain_numbering.py  --ignore_obsolete  auth2label  $DIR/family-auth.json  > $DIR/family.json
    # TODO decide about this branch based on config (not comment-out) (or just keep in dev)

    # Select random sample from the family
    print('\n::: SELECT SAMPLE :::')
    with RedirectIO(stdout=datadir.sub('sample.json')):
        select_random_domains.main(datadir.sub('family.json'), size=sample_size, or_all=conf.sample_selection.or_all, unique_pdb=conf.sample_selection.unique_pdb)
    with RedirectIO(stdout=datadir.sub('sample.simple.json')):
        simplify_domain_list.main(datadir.sub('sample.json'))
    sample_domains = lib_domains.load_domain_list(datadir.sub('sample.json'))
    n_sample = len(sample_domains)
    with datadir.sub('family_info.txt').open('a') as w:
        print('n_sample:', n_sample, file=w)
    
    # Download structures in CIF, cut the domains and save them as CIF and PDB
    print('\n::: DOWNLOAD :::')
    lib.run_dotnet(conf.download.structure_cutter_path, datadir.sub('sample.simple.json'), '--sources', ' '.join(conf.download.structure_sources), 
                   '--cif_outdir', datadir.sub('cif'), '--pdb_outdir', datadir.sub('pdb'), '--failures', datadir.sub('StructureCutter-failures.txt')) 
    
    # Check if some failed-to-download structures are obsolete or what; remove them from the sample if yes
    some_still_missing = remove_obsolete_pdbs.main(datadir.sub('sample.json'), datadir.sub('StructureCutter-failures.txt'), 
                                                   datadir.sub('sample-nonobsolete.json'), datadir.sub('StructureCutter-failures-nonobsolete.txt'))
    if some_still_missing != 0:
        raise Exception(f'Failed to download some structure and they are not obsolete. See {datadir.sub("StructureCutter-failures-nonobsolete.txt")}')
    else:
        datadir.sub('sample.json').mv(datadir.sub('sample-original.json'))
        datadir.sub('sample-nonobsolete.json').mv(datadir.sub('sample.json'))
        with RedirectIO(stdout=datadir.sub('sample.simple.json')):
            simplify_domain_list.main(datadir.sub('sample.json'))
    sample_domains = lib_domains.load_domain_list(datadir.sub('sample.json'))
    n_sample = len(sample_domains)
    with datadir.sub('family_info.txt').open('a') as w:
        print('n_sample_without_obsoleted:', n_sample, file=w)

    # Convert PDB and domain lists into various formats
    format_domains.main(datadir.sub('family.json'), datadir.sub('sample.json'), out_dir=datadir.sub('lists'))
    datadir.sub('family_info.txt').cp(datadir.sub('lists', 'family_info.txt'))
    if n_sample == 0:
        datadir.sub('EMPTY_FAMILY').clear()
        return 1

    # Perform multiple structure alignment by MAPSCI
    print('\n::: MAPSCI :::')
    run_mapsci.main(datadir.sub('sample.json'), datadir.sub('pdb'), datadir.sub('mapsci'), mapsci=conf.mapsci.mapsci_path, init=conf.mapsci.init, n_max=conf.mapsci.n_max)
    mapsci_consensus_to_cif.main(datadir.sub('mapsci', 'consensus.pdb'), datadir.sub('mapsci', 'consensus.cif'))

    # Align structures to the consensus backbone by PyMOL's CEalign
    print('\n::: CEALIGN :::')
    cealign_all.main(datadir.sub('mapsci', 'consensus.cif'), datadir.sub('sample.json'), datadir.sub('cif'), datadir.sub('cif_cealign'), progress_bar=True)

    if conf.files.clean_pdb_cif:
        datadir.sub('pdb').rm(recursive=True)
        datadir.sub('cif').rm(recursive=True)

    # Calculate SSE positions and cluster SSEs
    # Will add manual labels into the consensus annotation, if $DIR/manual-*.sses.json and $DIR/manual-*.cif exist
    print('\n::: CLUSTERING :::')
    lib_sses.compute_ssa(datadir.sub('sample.json'), datadir.sub('cif_cealign'), skip_if_exists = not conf.sec_str_consensus.force_ssa, 
                         secstrannotator_dll=conf.sec_str_consensus.secstrannotator_path, progress_bar=True)
    datadir.sub('sample.json').cp(datadir.sub('cif_cealign', 'sample.json'))
    with RedirectIO(tee_stdout=datadir.sub('making_guide_tree.log')):
        make_guide_tree.main(datadir.sub('cif_cealign'), show_tree=False, progress_bar=True)
    with RedirectIO(tee_stdout=datadir.sub('clustering.log')):
        acyclic_clustering.main(datadir.sub('cif_cealign'), force_ssa=conf.sec_str_consensus.force_ssa, secstrannotator_rematching= conf.sec_str_consensus.secstrannotator_rematching, min_occurrence=0, fallback=60)

    # Tidy up
    datadir.mkdir(results, exist_ok=True)
    for filename in str.split('lengths.tsv cluster_precedence_matrix.tsv statistics.tsv occurrence_correlation.tsv consensus.sses.json'):
        datadir.sub('cif_cealign', filename).cp(datadir.sub(results, filename))
    datadir.sub('mapsci', 'consensus.cif').cp(datadir.sub(results, 'consensus.cif'))

    # Draw diagrams
    print('\n::: DIAGRAM :::')
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram.svg'), json_output=datadir.sub(results, 'diagram.json'))
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_min20.svg'), occurrence_threshold=0.2)
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_conn.svg'), connectivity=True)
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_conn_min20.svg'), connectivity=True, occurrence_threshold=0.2)
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_conn_symcdf.svg'), connectivity=True, shape='symcdf')
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_conn_symcdf_min20.svg'), connectivity=True, shape='symcdf', occurrence_threshold=0.2)
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_conn_arrow_min20.svg'), connectivity=True, shape='arrow', occurrence_threshold=0.2)
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_conn_heatmap.svg'), connectivity=True, heatmap=True)
    draw_diagram.main(datadir.sub(results), dag=True, output=datadir.sub(results, 'diagram_conn_heatmap_min20.svg'), connectivity=True, heatmap=True, occurrence_threshold=0.2)
    # TODO convert SVG -> PNG

    # Visualize in PyMOL and save as a session + PNG
    print('\n::: VISUALIZE :::')
    lib_pymol.create_consensus_session(datadir.sub(results, 'consensus.cif'), datadir.sub(results, 'consensus.sses.json'), datadir.sub(results, 'consensus.pse'), 
                                       coloring=conf.visualization.coloring, out_image_file=datadir.sub(results, 'consensus.png'), image_size=(1000,))
    if conf.visualization.create_multi_session:
        lib_pymol.create_multi_session(datadir.sub('cif_cealign'), datadir.sub(results, 'consensus.cif'), datadir.sub(results, 'consensus.sses.json'), 
                                       datadir.sub(results, 'clustered.pse'), coloring=conf.visualization.coloring, progress_bar=True)

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

# #############################################################################################

