'''
Wrappers for shell-like functions from os, shutil...
(mv, cp, rm, ls...)
'''

from __future__ import annotations
from pathlib import Path
import shutil
from contextlib import suppress
from typing import Iterator


def mv(source: Path, dest: Path) -> Path:
    '''Move a file or directory ($ mv source dest)'''
    dest.parent.mkdir(parents=True, exist_ok=True)
    new_path: Path = shutil.move(source, dest)  # type: ignore
    return new_path

def cp(source: Path, dest: Path) -> Path:
    '''Copy a file or directory ($ cp -r source dest)'''
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        new_path = shutil.copytree(source, dest)
    else:
        new_path = shutil.copy(source, dest)
    return new_path

def rm(*paths: Path, recursive: bool = False, ignore_errors: bool = False) -> None:
    '''Remove this file or empty directory ($ rm path || rmdir path).
    If recursive, remove also non-empty directories ($ rm -r path).
    '''
    for path in paths:
        if ignore_errors:
            with suppress(OSError):
                rm(path, recursive=recursive, ignore_errors=False)
        else:
            if path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                else:
                    path.rmdir()
            else:
                path.unlink()

def ls(directory: Path, recursive: bool = False, only_files: bool = False, only_dirs: bool = False) -> list[Path]:
    '''List files in this directory or [] if self is not a directory. ($ ls [-R] directory)'''
    if recursive:
        result = list(_ls_recursive(directory))
    elif directory.is_dir():
        return sorted(directory.iterdir())
    else:
        raise OSError('directory must be a an existing directory')
    if only_files:
        result = [f for f in result if f.is_file()]
    if only_dirs:
        result = [f for f in result if f.is_dir()]
    return result

def _ls_recursive(path: Path, include_self: bool = False) -> Iterator[Path]:
    if include_self:
        yield path
    if path.is_dir():
        for file in ls(path):
            yield from _ls_recursive(file, include_self=True)


def archive(source: Path, dest: Path, rm_source: bool = False) -> Path:
    '''Create archive from a file or directory, 
    e.g. archive(Path('data'), Path('data.zip')) '''
    fmt = dest.suffix.lstrip('.')
    archive_name = dest.parent/dest.stem  # without suffix!
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(archive_name), fmt, str(source))
    if rm_source:
        rm(source, recursive=True)
    return Path(archive)
