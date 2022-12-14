# OverProt Core

**OverProt Core** is an algorithm that constructs the secondary
structure consensus for a given protein family.
The produced consensus can be used as a template for annotation of secondary structure elements in protein families, e.g. by SecStrAnnotator.

This file is focused on the information about how to run OverProt Core.
Detailed description of how it works can be found in `doc/Description_of_methods.pdf`.

OverProt Core is implemented mostly in Python3
and designed to run in the Linux environment (tested on Python3.8 on Ubuntu 20.04).
On other operating systems, it can be run in Docker (see **Execution via Docker** below).

## Installation

Before the first execution, the dependencies must be installed:

```sh
sh install.sh --clean
. venv/bin/activate
```

This installs the necessary Ubuntu packages, Python packages, .NET Core Runtime, and a free version of PyMOL. The remaining dependencies don't require installation and are simply stored in `dependencies/`: SecStrAnnotator (Midlik,A. et al. (2019) Automated family-wide annotation of secondary structure elements. Methods Mol Biol, 1958, 47–71. <https://sestra.ncbr.muni.cz/>), MAPSCI (Ilinkin,I. et al. (2010) Multiple structure alignment and consensus identification for proteins. BMC Bioinformatics, 11, 71. <http://www.geom-comp.umn.edu/mapsci>), StructureCutter (small utility created for OverProt).

In case you don't have the necessary permissions for installation (or it fails for a different reason), try running OverProt Core via Docker.

## Execution

All steps of the algorithm are combined in `overprot.py`.
It is run in a Python virtual environment.
Its arguments are the **CATH family ID** and the **output directory**:

```sh
. venv/bin/activate
python  overprot.py  --help
python  overprot.py  1.10.10.1020  data/1.10.10.1020/
```

This example will process all protein domains from the 1.10.10.1020 family and save the results into directory `data/1.10.10.1020/`.

```sh
python  overprot.py  1.10.630.10  data/cyp/  --sample_size 20
```

This example will process 20 random proteins from Cytochrome P450 (CYP) family (CATH code 1.10.630.10) and save the results into directory `data/cyp/`.

## Execution via Docker

In case you want to run OverProt Core in a Docker container, you can skip the **Installation** step above. Instead, make sure you have Docker correctly installed and you have permissions to run it.

In most Linux distributions, Docker is installed by the standard package management tools (e.g. on Ubuntu: `sudo apt-get install docker docker.io docker-compose`). You must also be a member of the `docker` group (`sudo usermod -aG docker $USER` and re-login).

In Windows and MacOS, you can install the Docker Desktop application (<https://www.docker.com/products/docker-desktop/)>.
You must start the Docker Desktop application and open a command line (`powershell`) to run the following steps.

Large scale compute environments, where the ordinary users don't have root permissions, typically provide some compatible Docker alternatives which allow running Docker containers without root permissions (e.g. Podman, Singularity).

1. Pull the Docker image from the repository:

   ```sh
   docker  pull  registry.gitlab.com/midlik/overprot/overprot-core
   docker  images  # Check that the image is downloaded
   ```

2. Start a container based on the image:

   ```sh
   docker  run  -it  -v /home/peppa/data:/data  registry.gitlab.com/midlik/overprot/overprot-core
   ```

   To make the OverProt results available outside of the container, the option `-v` mounts a directory on the host machine (here `/home/peppa/data`) to a directory in the container (here `/data`).

3. Within the container, run OverProtCore:

   ```sh
   python  overprot.py  --help
   python  overprot.py  1.10.630.10  /data/cyp/  --sample_size 20
   ```

   Thanks to the mounting, you will see the results in `/home/peppa/data/cyp/` on the host machine.

4. Once you're done, you can exit the container and discard it:

   ```sh
   exit
   docker  container  prune
   ```

Steps 2–4 can be combined to call OverProt without entering the container:

```sh
docker  run  -v /home/peppa/data:/data  registry.gitlab.com/midlik/overprot/overprot-core  -c "python overprot.py 1.10.630.10 /data/cyp/ --sample_size 20"
```

## Configuration

The default configuration is in `overprot-config.ini`. Each configuration item is also explained there.
You can change the configuration by copying this file, modifying the copy, and then running OverProt with `--config` option.

```sh
python  overprot.py  1.10.630.10  data/cyp/  --sample_size 20  --config overprot-config-customized.ini
```

If you run OverProt Core in a Docker container, remember that all changes you do within the container will be discarded with the container itself (except for the mounted files). However, you can mount a customized configuration file into the container:

```sh
docker  run  -v /home/peppa/data:/data  -v /home/peppa/overprot-config-customized.ini:/overprot-config-customized.ini  registry.gitlab.com/midlik/overprot/overprot-core  -c "python overprot.py 1.10.630.10 /data/cyp/ --sample_size 20  --config /overprot-config-customized.ini"
```

## Running on custom datasets

Normally, OverProt Core takes a CATH ID of a family and downloads the list of its domains. Another option is to provide a custom list of domains.

Prepare the list:

`data/my_domains.txt`:

```text
1og2,A 
1og2,B 
1bu7,A,100:450 
1bu7,B,100:178,185:370,390:
```

Run OverProt Core:

```sh
python  overprot.py  -  data/custom_family/  --domains data/my_domains.txt
```

The structures don't need to be in the PDB database. In such case, you must provide them in mmCIF format and configure the `structure_sources` setting accordingly:

`data/my_domains2.txt`:

```text
struct1,A
struct2,A
struct3,B,100:200
```

`overprot-config-customized.ini`:

```ini
...
[download]
structure_sources = file:///path/to/my/structures/{pdb}.cif
...
```

Run OverProt Core:

```sh
python  overprot.py  -  data/custom_family2/ --domains data/my_domains2.txt --config overprot-config-customized.ini
```

(OverProt will replace `{pdb}` by the individual structure names (`struct1`, `struct2`, `struct3`) to get the names of the input structure files (`/path/to/my/structures/struct1.cif` etc.).
An alternative to modifying the config file is to use the `--structure_source` option.)

## Steps

OverProt Core algorithm performs the following steps.
Detailed description of each step is provided in `doc/Description_of_methods.pdf`.
Individual steps are submodules located in `overprot/` and can be run separately by:

```sh
python  -m overprot.{submodule}  --help
```

- **Download the list of domains** for the family (by `overprot.domains_from_pdbeapi`).
Output:
  - `family.json`

- **Select random sample of domains** (by `overprot.select_random_domains`). By default selects max. one domain per PDB entry (changed by `[sample_selection]unique_pdb` in the config file).
Output:
  - `sample.json`

- **Download selected structures**, cut the domains, save in mmCIF (by `dependencies/StructureCutter`). Also saves the structures with renumbered residues in PDB format (to be used by `MAPSCI`).
Output:
  - `cif/`
  - `pdb/`

- **Convert the lists of PDB entries and domains into various formats** (by `overprot.format_domains`)
Output:
  - `lists/`

- **Multiple structure alignment** to produce consensus structure by `MAPSCI` (by `overprot.run_mapsci` and `overprot.mapsci_consensus_to_cif`).
Output:
  - `mapsci/`
  - `mapsci/consensus.cif`

- **Align structures to consensus structure** (by `overprot.cealign_all`).
Output:
  - `cif_cealign/`

- **Secondary structure assignment** - detect the SSEs in all domains (by `overprot.libs.lib_sses.compute_ssa()`).
Output:
  - `cif_cealign/*.sses.json`

- **Create the guide tree** (by `overprot.make_guide_tree`).
Output:
  - `cif_cealign/guide_tree.newick`
  - `cif_cealign/guide_tree.children.tsv`

- **Cluster the SSEs** - the main part (by `overprot.acyclic_clustering`).
Output:
  - `cif_cealign/*-clust.sses.json`
  - `results/consensus.sses.json` - generated SSE consensus (1 cluster = 1 consensus SSEs)
  - `results/statistics.tsv`
  - `results/lengths.tsv`
  - `results/cluster_precedence_matrix.tsv`
  - `results/occurrence_correlation.tsv`

- **Draw 1D diagrams** (by `overprot.draw_diagram`).
Output:
  - `results/diagram*.svg` - order of SSEs shown by directed acyclic graph, (gray = helices, colors = sheets)
  - `results/diagram.json` - preprocessed data for interactive visualization by OverProt Viewer

- **Annotate whole family** - this is only done if `[annotation]annotate_whole_family` is `True` in the config file. (Caution: this annotates ALL members of the family even if only a subset was used for consensus generation (`--sample_size`)) (by `overprot.libs.lib_sses.annotate_all_with_SecStrAnnotator()`).
Output:
  - `annotated_sses/`

- **Visualize in PyMOL** (by `overprot.libs.lib_pymol.create_consensus_session()` and `overprot.libs.lib_pymol.create_multi_session()`).
Output:
  - `results/consensus.pse` - consensus structure with generated SSE consensus (cylinders = helices, arrows = strands, width = occurrence)
  - `results/clustered.pse` - consensus + all structures and their clustered SSEs (time consuming, only done if `[visualization]create_multi_session` is `True`)

## Multi-family execution

Multiple families can be processed in parallel using `overprot_multifamily.py`.
Its arguments are the **family list** and the **output directory**:

```sh
python  overprot_multifamily.py  --help
python  overprot_multifamily.py  data/families.txt  data/multifamily/
```

It is also possible to download the list of all CATH families automatically and collect the result files by type:

```sh
python  overprot_multifamily.py  -  data/multifamily/  --download_family_list  --collect
```
