'''
Logging utilities (progress bar, timing)
'''

from __future__ import annotations
import sys
import shutil
from datetime import datetime
from typing import TextIO, Optional, Literal


class Timing:
    def __init__(self, name: Optional[str] = None, file: TextIO|Literal['stdout', 'stderr'] = 'stdout', mute=False):
        self.name = name
        self.file = sys.stdout if file == 'stdout' else sys.stderr if file == 'stderr' else file
        self.mute = mute
        self.time = None
    def __enter__(self):
        self.t0 = datetime.now()
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        dt = datetime.now() - self.t0
        self.time = dt
        if not self.mute:
            message = f'Timing:\t{dt}\t'
            if exc_type is not None:
                message += '[Failed] '
            if self.name is not None:
                message += self.name
                # message = f'Timing:\t{dt}\t{self.name}'
            # else:
                # message = f'Timing:\t{dt}'
            print(message, file=self.file)


class ProgressBar:
    DONE_SYMBOL = '█'
    TODO_SYMBOL = '-'

    def __init__(self, n_steps: int, *, width: Optional[int] = None, 
                 title: str = '', prefix: Optional[str] = None, suffix: Optional[str] = None, 
                 writer: TextIO|Literal['stdout', 'stderr'] = 'stdout', mute: bool = False, timing: bool = True):
        self.n_steps = n_steps # expected number of steps
        self.prefix = prefix + ' ' if prefix is not None else ''
        self.suffix = ' ' + suffix if suffix is not None else ''
        self.writer = sys.stdout if writer == 'stdout' else sys.stderr if writer == 'stderr' else writer  # writer if writer is not None else sys.stdout
        self.width = width or _get_width(self.writer)  # shutil.get_terminal_size().columns
        self.width -= len(self.prefix) + len(self.suffix) + 10
        self.title = (' '+title+' ')[0:min(len(title)+2, self.width)]
        self.done = 0 # number of completed steps
        self.shown = 0 # number of shown symbols
        self.mute = mute
        self.timing = Timing(title, file=writer) if timing and not mute else None

    def __enter__(self):
        if not self.mute:
            self.writer.write(' ' * len(self.prefix))
            self.writer.write('┌' + self.title + '─'*(self.width-len(self.title)) + '┐\n')
            self.writer.flush()
            self.step(0, force=True)
        if self.timing is not None:
            self.timing.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        # self.finalize(completed = exc_type is None)
        completed = exc_type is None
        if not self.mute:
            if completed:
                self.step(self.n_steps - self.done)
            self.writer.write('\n')
            self.writer.flush()
        if self.timing is not None:
            self.timing.__exit__(exc_type, exc_value, exc_traceback)

    # def start(self):
    #     if not self.mute:
    #         self.writer.write(' ' * len(self.prefix))
    #         self.writer.write('┌' + self.title + '─'*(self.width-len(self.title)) + '┐\n')
    #         self.writer.flush()
    #         self.step(0, force=True)
    #     if self.timing is not None:
    #         self.timing.__enter__()
    #     return self

    def step(self, n_steps=1, force=False):
        if not self.mute:
            if n_steps == 0 and not force:
                return
            self.done = min(self.done + n_steps, self.n_steps)
            try:
                progress = self.done / self.n_steps
            except ZeroDivisionError:
                progress = 1.0
            new_shown = int(self.width * progress)
            if new_shown != self.shown or force:
                self.writer.write(f'\r{self.prefix}└')
                self.writer.write(self.DONE_SYMBOL * new_shown + self.TODO_SYMBOL * (self.width - new_shown))
                self.writer.write(f'┘ {int(100*progress):>3}%{self.suffix} ')
                self.writer.flush()
                self.shown = new_shown  

    # def finalize(self, completed=True):
    #     if not self.mute:
    #         if completed:
    #             self.step(self.n_steps - self.done)
    #         self.writer.write('\n')
    #         self.writer.flush()
    #     if self.timing is not None:
    #         self.timing.__exit__()


def _get_width(writer: TextIO) -> int:
    if writer.isatty():
        return shutil.get_terminal_size().columns
    else:
        return 80
