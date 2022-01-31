'''
Logging utilities (progress bar, timing)
'''

from __future__ import annotations
import sys
import shutil
from datetime import datetime
from typing import TextIO, Optional, Literal


class Timing:
    '''Context manager which measures time of the block execution.
    @param  `name`  Name used when printing measured time
    @param  `file`  Output stream for printing measure time or 'stdout' (default) or 'stderr'
    @param  `mute`  Supress all printing (measured time can still be accessed by .time)
    '''
    def __init__(self, name: str = '', file: TextIO|Literal['stdout', 'stderr'] = 'stdout', mute=False):
        self.name = name
        self.file = sys.stdout if file == 'stdout' else sys.stderr if file == 'stderr' else file
        self.mute = mute
        self.time = None
    def __enter__(self):
        self.t0 = datetime.now()
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.time = datetime.now() - self.t0
        if not self.mute:
            err_flag = '[Failed] ' if exc_type is not None else ''
            message = f'Timing:\t{self.time}\t{err_flag}{self.name}'
            print(message, file=self.file)


class ProgressBar:
    '''Context manager which prints current progress of an iterative process (number of steps must be known before).
    Completed steps are reported to ProgressBar by .step()
    @param  `n_step`  Number of step to be iterated through
    @param  `width`   Width of the bar in textual output (number of characters, default: console width or 80)
    @param  `writer`  Output stream for printing the bar or 'stdout' (default) or 'stderr'
    @param  `mute`    Supress all printing
    @param  `timing`  Also time the whole process and print time at the end
    '''
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
        completed = exc_type is None
        if not self.mute:
            if completed:
                self.step(self.n_steps - self.done)
            self.writer.write('\n')
            self.writer.flush()
        if self.timing is not None:
            self.timing.__exit__(exc_type, exc_value, exc_traceback)

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


def _get_width(writer: TextIO) -> int:
    if writer.isatty():
        return shutil.get_terminal_size().columns
    else:
        return 80
