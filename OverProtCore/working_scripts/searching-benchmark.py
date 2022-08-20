from overprot.libs.lib_logging import Timing
from pathlib import Path

from searching31 import Searcher

with Timing():
    DATA = Path('/home/adam/Workspace/Python/OverProt/docker_mount/data/db/domain_list.csv')

    searcher = Searcher(DATA)

results = searcher.get_domains_families_for_pdb('5w97')
# for dom, fam in results:
#     print(dom, fam)
print(len(results))
print(searcher.search_domain('5w97h00'))
print(searcher.search_pdb('1TQN'))

# v1 ~ 2.58  (original)
# v2 ~ 2.50  (_make_doms_fams_range_dict) (baseline 2.21 = no pdb_to_doms_fams)
# v3 ~ 2.10  (_BaseReprManager with ranges)  / 1.77 without pdb_to_doms_fams / 1.31 with BaseReprManager
# v4 ~ 2.14  (_BaseReprManager with index and offsets) / 1.81 without pdb_to_doms_fams   ----> not good
# v3.1 ~ 1.87  (only one dict for domains)  / 1.59  
# --> out of this, 0.72 is in sorts --> try sort in numpy?
# --> try to drop _domain_to_fam_pdb_chain_ranges? (can be read from files) -> would get to 1.72


