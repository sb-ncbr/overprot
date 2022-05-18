'''
Locations of dependencies.
'''

from __future__ import annotations
import sys
from pathlib import Path
import subprocess
from typing import List


def _find_dotnet(candidates: List[str|Path], version: str) -> str|Path:
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, '--list-runtimes'], check=True, capture_output=True, encoding='utf8')# stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            assert version in result.stdout, f'Dotnet candidate "{candidate}" does not have the correct runtime version ({version})'
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError, AssertionError):
            pass
    raise FileNotFoundError(f'Cannot find dotnet, version {version} (tried these: {candidates})')


PYTHON = sys.executable
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DEFAULT_CONFIG_FILE = PROJECT_ROOT_DIR/'overprot-config.ini'
STRUCTURE_CUTTER_DLL = PROJECT_ROOT_DIR/'dependencies'/'StructureCutter'/'bin'/'Release'/'net6.0'/'StructureCutter.dll'
MAPSCI_EXE = PROJECT_ROOT_DIR/'dependencies'/'mapsci-1.0'/'bin'/'mapsci'
SECSTRANNOTATOR_DLL = PROJECT_ROOT_DIR/'dependencies'/'SecStrAnnotator'/'SecStrAnnotator.dll'
SECSTRANNOTATOR_BATCH_PY = PROJECT_ROOT_DIR/'dependencies'/'SecStrAnnotator'/'SecStrAnnotator_batch.py'
DOTNET_VERSION = 'Microsoft.NETCore.App 6.0'
DOTNET = _find_dotnet(['dotnet', Path.home()/'.dotnet'/'dotnet'], DOTNET_VERSION)
