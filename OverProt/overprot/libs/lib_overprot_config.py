from typing import List, Literal

from .lib_config import ConfigSection, Config


class OverProtConfig(Config):
    class _DownloadSection(ConfigSection):
        structure_sources: List[str]
    class _SampleSelectionSection(ConfigSection):
        unique_pdb: bool
        or_all: bool
    class _MapsciSection(ConfigSection):
        init: Literal['median', 'center']
        n_max: int
    class _OverProtSection(ConfigSection):
        force_ssa: bool
        secstrannotator_rematching: bool
        annotate_whole_family: bool
    class _FilesSection(ConfigSection):
        results_dir: str
        clean_pdb_cif: bool
        clean_aligned_cif: bool
    class _VisualizationSection(ConfigSection):
        coloring: Literal['color', 'rainbow']
        create_multi_session: bool

    download: _DownloadSection
    sample_selection: _SampleSelectionSection
    mapsci: _MapsciSection
    overprot: _OverProtSection
    files: _FilesSection
    visualization: _VisualizationSection