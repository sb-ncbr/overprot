'''
Align all domains listed in `sample_file` to the structure in `target_file`.

Example usage:
    python3  -m overprot.cealign_all  --help
'''

from pathlib import Path
from typing import Optional

from .libs import lib_domains
from .libs import lib_pymol
from .libs.lib_cli import cli_command, run_cli_command

#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

@cli_command()
def main(target_file: Path, sample_file: Path, in_directory: Path, out_directory: Path, progress_bar: bool = False) -> Optional[int]:
    '''Align all domains listed in `sample_file` to the structure in `target_file`.
    @param  `target_file`    CIF file with target structure.
    @param  `sample_file`    JSON file with list of domains to be aligned.
    @param  `in_directory`   Directory with input structures (named {domain_name}.cif).
    @param  `out_directory`  Directory for output structures (named {domain_name}.cif).
    @param  `progress_bar`   Show progress bar.
    '''
    domains = lib_domains.load_domain_list(sample_file)
    mobiles = [dom.name for dom in domains]
    out_directory.mkdir(parents=True, exist_ok=True)
    mobile_files = [in_directory/f'{mobile}.cif' for mobile in mobiles]
    result_files = [out_directory/f'{mobile}.cif' for mobile in mobiles]
    result_ttt_files = [out_directory/f'{mobile}-ttt.csv' for mobile in mobiles]
    lib_pymol.cealign_many(target_file, mobile_files, result_files, ttt_files=result_ttt_files, fallback_to_dumb_align=True, show_progress_bar=progress_bar)
    return None


if __name__ == '__main__':
    run_cli_command(main)
