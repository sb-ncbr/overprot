# OverProt

# TODO update

The goal of OverProt is automatic generation of SSE (secondary structure elements) annotation templates.

# How to run

The main script is `ubertemplate.sh`.

Example:

`bash ubertemplate.sh 1.10.630.10 50 ../data/cyp_50/` will process 50 random proteins from CYP family (CATH code 1.10.630.10) and save the results into directory `../data/cyp_50/`.


# Steps

The following steps are performed. (Individual steps can be skipped by commenting out corresponding sections in `ubertemplate.sh`.)

  * **Download the list of domains** for the family (by `domains_from_pdbeapi.py`).
  Output:
    * `family.json`

  * **Select random sample of domains** (by `select_random_domains.py`). By default selects max. one domain per PDB entry (`--unique_pdb`).
  Output:
    * `sample.json`

  * **Download selected structures**, cut the domains, save in CIF (by `StructureCutter`). Also saves the structures with renumbered residues in PDB (to be used by `MAPSCI`).
  Output:
    * `cif/`
    * `pdb/`

  * **Multiple structure alignment** to produce consensus structure (by `MAPSCI`).
  Output:
    * `pdb_mapsci/`
    * `consensus.cif`

  * **Align structures to consensus structure** (by `cealign_all.py`).
  Output:
    * `cif_cealign/`

  * **Cluster the SSEs** - the main part (by `acyclic_clustering_sides.py`).
  Output:
    * `cif_cealign/*-clust.sses.json`
    * `results/consensus.sses.json` - generated ubertemplate (1 cluster = 1 ubertemplate SSEs)
    * `results/statistics.tsv`
    * `results/lengths.tsv`
    * `results/cluster_precedence_matrix.tsv`
    * `results/occurrence_correlation.tsv`

  * **Draw 1D diagram** (by `draw_diagram.py`).
  Output:
    * `results/diagram_dag.svg` - order of SSEs shown by directed acyclic graph, (gray = helices, colors = sheets)
    * `results/diagram.svg` - order of SSEs simplified to a sequence

  * **Visualize in PyMOL** (by `load_clustered_sses.py`).
  Output:
    * `results/consensus.pse` - consensus structure with generated ubertemplate (flat ends = helices, round ends = strands, width = occurrence)
    * `results/clustered.pse` - consensus + all structures and their clustered SSEs