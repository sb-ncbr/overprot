'''
Run all steps of the OverProt algorithm.

Example usage:
    python3  -m overprot.oveprot  1.10.10.1020  data/1.10.10.1020  --sample_size all
'''

from __future__ import annotations
import sys
import contextlib
from pathlib import Path
from typing import Optional

from .libs import lib
from .libs import lib_sh
from .libs import lib_run
from .libs import lib_domains
from .libs import lib_sses
from .libs import lib_pymol
from .libs.lib_logging import Timing
from .libs.lib_io import RedirectIO
from .libs.lib_overprot_config import OverProtConfig, ConfigException
from .libs.lib_pipeline import Pipeline, PipelineStepSpecificationError
from .libs.lib_cli import cli_command, run_cli_command, CliCommandExit

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


EXIT_CODE_MISSING_CONFIG = 3
EXIT_CODE_WRONG_CONFIG = 4
EXIT_CODE_EMPTY_FAMILY = 5

STEPS_HELP = '''Option --steps can be used to define which steps of the whole pipeline should be run.
Its value is a comma-separated list of items, where each item is either a step name, or a range of steps, e.g. 'A:B+'.
The + symbol after a step name instructs to start/end after that step, without + it starts/ends before that step.
E.g. --steps ':D, G+:K, P:R+' will run steps A, B, C, H, I, J, P, Q, R.
--steps '?' will print the list of all available steps, without actually running any steps.
The following is the list of all available steps.
'''

def handle_step_spec_error(error: PipelineStepSpecificationError) -> None:
    print('ERROR:', error, file=sys.stderr)
    exit()

@cli_command(parsers={'sample_size': lib.int_or_all})
def main(family: str, outdir: Path, 
         sample_size: Optional[int] = None, config: Optional[Path] = DEFAULT_CONFIG_FILE, # TODO allow X|None syntax for @cli_command
         domains: Optional[Path] = None, structure_source: Optional[str] = None, steps: str = ':',
         out: Optional[Path] = None, err: Optional[Path] = None) -> int:
    '''Run all steps of the OverProt algorithm.
    @param  `family`            Family identifier for PDBe API (e.g. CATH code).
    @param  `outdir`            Directory for results.
    @param  `sample_size`       Number of domains to process (integer or "all"). [default: "all"]
    @param  `config`            Configuration file.
    @param  `domains`           File with the list of input domains (do not download domain list).
    @param  `structure_source`  Prepend a structure source to download.structure_sources parameter from the config file.
    @param  `steps`             Select which steps of the whole pipeline should be run. Use --steps ? to print more info.
    @param  `out`               File for stdout.
    @param  `err`               File for stderr.
    '''
    if config is None:
        config = DEFAULT_CONFIG_FILE
    try:
        conf = OverProtConfig(config)
    except OSError:
        raise CliCommandExit(EXIT_CODE_MISSING_CONFIG, f'Cannot open configuration file: {config}')
    except ConfigException as ex:
        raise CliCommandExit(EXIT_CODE_WRONG_CONFIG, str(ex))

    if structure_source is not None and structure_source != '':
        conf.download.structure_sources.insert(0, structure_source)
    results = conf.files.results_dir
    sample_for_annotation = outdir/'sample-whole_family.json' if conf.annotation.annotate_whole_family else outdir/'sample.json'
    
    with RedirectIO(stdout=out, stderr=err), Timing('Total'):

        with Pipeline('OVERPROT_PIPELINE', steps=steps, help=STEPS_HELP, step_spec_error_handler=handle_step_spec_error) as pipeline:

            @pipeline.always
            def PRINT_CONFIG():
                '''Print configuration file and output directory'''
                print('Configuration:', config)
                print('Output directory:', outdir)

            @pipeline.step
            def GET_DOMAIN_LIST():
                '''Download the list of domains in the family'''
                with Timing('Get domain list'):
                    outdir.mkdir(parents=True, exist_ok=True)
                    with open(outdir/'family_info.txt', 'w') as w:
                        print('family_id:', family, file=w)
                    if domains is None:
                        with contextlib.redirect_stderr(sys.stdout):
                            with RedirectIO(stdout=outdir/'family-orig.json'):
                                domains_from_pdbeapi.main(family, join_domains_in_chain=True)
                    else:
                        lib_domains.save_domain_list(lib_domains.load_domain_list_by_pdb(domains), outdir/'family-orig.json', by_pdb=True)
                    domains_by_pdb = lib_domains.load_domain_list_by_pdb(outdir/'family-orig.json')
                    n_pdbs = len(domains_by_pdb)
                    n_domains = sum(len(doms) for doms in domains_by_pdb.values())
                    with open(outdir/'family_info.txt', 'a') as w:
                        print('n_pdbs:', n_pdbs, file=w)
                        print('n_domains:', n_domains, file=w)

                    lib_sh.cp(outdir/'family-orig.json', outdir/'family.json')
                    # python3  domains_with_observed_residues.py  --min_residues 4  $DIR/family-orig.json  >  $DIR/family.json  # TODO solve without an API call for each PDB
        
            # # Create a list of domains based on DALI results
            # echo '::: PREPARE DOMAIN LIST :::'
            # mkdir  -p  $DIR
            # python3  dali_hits_to_domains.py  --max_loop 50  $DIR/DALI.txt  > $DIR/family-auth.json
            # python3  convert_domain_numbering.py  --ignore_obsolete  auth2label  $DIR/family-auth.json  > $DIR/family.json
            # TODO decide about this branch based on config (not comment-out) (or just keep in dev)

            @pipeline.step
            def SELECT_SAMPLE():
                '''Select a random sample from the family'''
                with Timing('Select sample'):
                    with contextlib.redirect_stderr(sys.stdout):
                        with RedirectIO(stdout=outdir/'sample.json'):
                            select_random_domains.main(outdir/'family.json', size=sample_size, or_all=conf.sample_selection.or_all, unique_pdb=conf.sample_selection.unique_pdb)
                    if conf.annotation.annotate_whole_family:
                        with RedirectIO(stdout=outdir/'sample-whole_family.json'):
                            select_random_domains.main(outdir/'family.json', size=None, unique_pdb=False)
                    n_sample = len(lib_domains.load_domain_list(outdir/'sample.json'))
                    with open(outdir/'family_info.txt', 'a') as w:
                        print('n_sample:', n_sample, file=w)
        
            @pipeline.step
            def DOWNLOAD():
                '''Download structures in CIF, cut the domains, and save them as CIF and PDB'''
                with Timing('Download structures'):
                    lib_run.run_dotnet(STRUCTURE_CUTTER_DLL, sample_for_annotation, '--sources', ' '.join(conf.download.structure_sources), 
                                '--cif_outdir', outdir/'cif', '--pdb_outdir', outdir/'pdb', 
                                '--failures', outdir/'StructureCutter-failures.txt', 
                                timing=True) 
        
            @pipeline.step
            def CHECK_MISSING():
                '''Check if some failed-to-download structures are obsolete; remove them from the sample if yes'''
                some_still_missing = remove_obsolete_pdbs.main(outdir/'sample.json', outdir/'StructureCutter-failures.txt', 
                                                            outdir/'sample-nonobsolete.json', outdir/'StructureCutter-failures-nonobsolete.txt')
                if some_still_missing != 0:
                    raise Exception(f'Failed to download some structure and they are not obsolete. See {outdir/"StructureCutter-failures-nonobsolete.txt"}')
                else:
                    lib_sh.mv(outdir/'sample.json', outdir/'sample-original.json')
                    lib_sh.mv(outdir/'sample-nonobsolete.json', outdir/'sample.json')
                n_sample = len(lib_domains.load_domain_list(outdir/'sample.json'))
                with open(outdir/'family_info.txt', 'a') as w:
                    print('n_sample_without_obsoleted:', n_sample, file=w)

            @pipeline.step
            def MAKE_LISTS():
                '''Convert PDB entry and domain lists into various formats in lists/ subdirectory'''
                format_domains.main(outdir/'family.json', outdir/'sample.json', out_dir=outdir/'lists', per_domain_out_dir=outdir/'domain_info', family_id=family)
                lib_sh.cp(outdir/'family_info.txt', outdir/'lists'/'family_info.txt')
                n_sample = len(lib_domains.load_domain_list(outdir/'sample.json'))
                if n_sample == 0:
                    (outdir/'EMPTY_FAMILY').write_text('')
                    print('Cannot continue because the family is empty.')
                    raise CliCommandExit(EXIT_CODE_EMPTY_FAMILY, 'Cannot continue because the family is empty.')

            @pipeline.step
            def MAPSCI():
                '''Run multiple structure alignment by MAPSCI'''
                with Timing('MAPSCI'):
                    run_mapsci.main(outdir/'sample.json', outdir/'pdb', outdir/'mapsci', init=conf.mapsci.init, n_max=conf.mapsci.n_max)
                    mapsci_consensus_to_cif.main(outdir/'mapsci'/'consensus.pdb', outdir/'mapsci'/'consensus.cif', smooth_output_cif=outdir/'mapsci'/'consensus-smooth.cif')

            @pipeline.step
            def CEALIGN():
                '''Align structures to the MAPSCI consensus backbone by PyMOL's CEalign'''
                cealign_all.main(outdir/'mapsci'/'consensus.cif', sample_for_annotation, outdir/'cif', outdir/'cif_cealign', progress_bar=True)
                lib_sh.cp(outdir/'mapsci'/'consensus.cif', outdir/'cif_cealign'/'consensus.cif')

            @pipeline.step
            def CLUSTER():
                '''Calculate SSE positions and cluster SSEs'''
                with Timing('Clustering - total'):
                    lib_sses.compute_ssa(sample_for_annotation, outdir/'cif_cealign', skip_if_exists = not conf.overprot.force_ssa, progress_bar=True)
                    lib_sh.cp(outdir/'sample.json', outdir/'cif_cealign'/'sample.json')
                    with RedirectIO(tee_stdout=outdir/'making_guide_tree.log'):
                        make_guide_tree.main(outdir/'cif_cealign', show_tree=False, progress_bar=True)
                    with RedirectIO(tee_stdout=outdir/'clustering.log'):
                        acyclic_clustering.main(outdir/'cif_cealign', secstrannotator_rematching=conf.overprot.secstrannotator_rematching, fallback=30)
        
            @pipeline.step
            def ANNOTATE():
                '''Annotate all family members (if [annotation]annotate_whole_family is True)'''
                if conf.annotation.annotate_whole_family:
                    domains_by_pdb = lib_domains.load_domain_list_by_pdb(outdir/'family-orig.json')
                    with Timing('Annotation with SecStrAnnotator'):
                        lib_sses.annotate_all_with_SecStrAnnotator([dom for doms in domains_by_pdb.values() for dom in doms], outdir/'cif_cealign', 
                                                                    extra_options='--fallback 30 --unannotated', occurrence_threshold=conf.annotation.occurrence_threshold, outdirectory=outdir/'annotated_sses')
            
            @pipeline.step
            def TIDY_UP():
                '''Copy important result files into results/ subdirectory'''
                (outdir/results).mkdir(parents=True, exist_ok=True)
                for filename in str.split('consensus.cif consensus-smooth.cif'):
                    lib_sh.cp(outdir/'mapsci'/filename, outdir/results/filename)
                for filename in str.split('lengths.tsv cluster_precedence_matrix.tsv statistics.tsv occurrence_correlation.tsv consensus.sses.json'):
                    lib_sh.cp(outdir/'cif_cealign'/filename, outdir/results/filename)
        
            @pipeline.step
            def DIAGRAMS():
                '''Draw diagrams'''
                with Timing('Draw diagrams'):
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram.svg', json_output=outdir/results/'diagram.json')
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_min20.svg', occurrence_threshold=0.2)
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_conn.svg', connectivity=True)
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_conn_min20.svg', connectivity=True, occurrence_threshold=0.2)
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_conn_symcdf.svg', connectivity=True, shape='symcdf')
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_conn_symcdf_min20.svg', connectivity=True, shape='symcdf', occurrence_threshold=0.2)
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_conn_arrow_min20.svg', connectivity=True, shape='arrow', occurrence_threshold=0.2)
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_conn_heatmap.svg', connectivity=True, heatmap=True)
                    draw_diagram.main(outdir/results, dag=True, output=outdir/results/'diagram_conn_heatmap_min20.svg', connectivity=True, heatmap=True, occurrence_threshold=0.2)

            @pipeline.step
            def VISUALIZE():
                '''Visualize in PyMOL and save as a PSE session + PNG'''
                with Timing('Visualize'):
                    consensus_struct = outdir/results/'consensus-smooth.cif' if conf.visualization.use_smooth_trace else outdir/results/'consensus.cif'
                    lib_pymol.create_consensus_session(consensus_struct, outdir/results/'consensus.sses.json', outdir/results/'consensus.pse', 
                                                    coloring=conf.visualization.coloring, out_image_file=outdir/results/'consensus.png', image_size=(1000,))
                    if conf.visualization.create_multi_session:
                        lib_pymol.create_multi_session(outdir/'cif_cealign', consensus_struct, outdir/results/'consensus.sses.json', 
                                                    outdir/results/'clustered.pse', coloring=conf.visualization.coloring, progress_bar=True)
            

            @pipeline.step
            def CLEAN():
                '''Remove PDB and CIF files (if [files]clean_pdb_cif is True), remove aligned CIF files (if [files]clean_aligned_cif is True)'''
                if conf.files.clean_pdb_cif:
                    lib_sh.rm(outdir/'pdb', recursive=True, ignore_errors=True)
                    lib_sh.rm(outdir/'cif', recursive=True, ignore_errors=True)
                if conf.files.clean_aligned_cif:
                    lib_sh.rm(*(outdir/'cif_cealign').glob('*.cif'))
        
            @pipeline.always
            def COMPLETED():
                '''Print happy-ending message'''

        # TODO write docs for each module/script, update README.txt
    return 0


def _main():
    run_cli_command(main)


if __name__ == '__main__':
    _main()



# TODO: create sample files (JSON...) in a directory and reference them in README.md and in docstrings
# TODO: prune code and remove unnecessary settings
# TODO: document important library functions
