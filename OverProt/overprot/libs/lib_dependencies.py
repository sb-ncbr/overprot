'''Library of locations of dependencies'''

import sys
from pathlib import Path
import subprocess
from typing import List


def _find_dotnet(candidates: List[str]) -> str:
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, '--info'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    raise FileNotFoundError(f'Cannot find dotnet (tried these: {candidates})')


PYTHON = sys.executable
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DEFAULT_CONFIG_FILE = str(PROJECT_ROOT_DIR/'overprot-config.ini')
STRUCTURE_CUTTER_DLL = str(PROJECT_ROOT_DIR/'dependencies'/'StructureCutter'/'bin'/'Release'/'netcoreapp3.1'/'StructureCutter.dll')
MAPSCI_EXE = str(PROJECT_ROOT_DIR/'dependencies'/'mapsci-1.0'/'bin'/'mapsci')
SECSTRANNOTATOR_DLL = str(PROJECT_ROOT_DIR/'dependencies'/'SecStrAnnotator'/'SecStrAnnotator.dll')
SECSTRANNOTATOR_BATCH_PY = str(PROJECT_ROOT_DIR/'dependencies'/'SecStrAnnotator'/'SecStrAnnotator_batch.py')
DOTNET = _find_dotnet(['dotnet', str(Path.home()/'.dotnet'/'dotnet')])
