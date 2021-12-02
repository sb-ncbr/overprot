'''
Running external programs in subprocesses.
'''

import sys
import subprocess
from pathlib import Path
from typing import Optional

from .lib_logging import Timing
from .lib_io import maybe_open, consolidate_string
from .lib_dependencies import DOTNET


def run_command(*args, stdin: Optional[str] = None, stdout: Optional[str] = None, stderr: Optional[str] = None, 
                appendout: bool = False, appenderr: bool = False, timing: bool = False) -> int:
    out_mode = 'a' if appendout else 'w'
    err_mode = 'a' if appenderr else 'w'
    arglist = list(map(str, args))
    with maybe_open(stdin, 'r', default=sys.stdin) as stdin_handle:
        with maybe_open(stdout, out_mode, default=sys.stdout) as stdout_handle: 
            with maybe_open(stderr, err_mode, default=sys.stderr) as stderr_handle:
                normal_streams = hasattr(stdout_handle, 'fileno') and hasattr(stderr_handle, 'fileno')

                with Timing(f'Run command "{" ".join(arglist)} ..."', mute = not timing):
                    if normal_streams:
                        process = subprocess.run(arglist, check=True, stdin=stdin_handle, stdout=stdout_handle, stderr=stderr_handle)
                    else:
                        process = subprocess.run(arglist, check=True, stdin=stdin_handle, capture_output=True)
                        stdout_handle.write(consolidate_string(process.stdout.decode('utf8')))
                        stderr_handle.write(consolidate_string(process.stderr.decode('utf8')))
    return process.returncode

def run_dotnet(dll: Path, *args, **run_command_kwargs):
    if not dll.is_file():  # dotnet returns random exit code, if the DLL is not found ¯\_(ツ)_/¯
        raise FileNotFoundError(dll)
    run_command(DOTNET, dll, *args, **run_command_kwargs)
