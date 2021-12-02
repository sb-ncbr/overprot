'''
Manipulation with IO operations (redirecting stdout, stderr)...
'''

import sys
from pathlib import Path
from contextlib import contextmanager
from typing import TextIO, Optional


class RedirectIO:
    def __init__(self, stdin: Optional[Path] = None, stdout: Optional[Path] = None, stderr: Optional[Path] = None, 
                 tee_stdout: Optional[Path] = None, tee_stderr: Optional[Path] = None, 
                 append_stdout: bool = False, append_stderr: bool = False):
        assert stdout is None or tee_stdout is None, f'Cannot specify both stdout and tee_stdout'
        assert stderr is None or tee_stderr is None, f'Cannot specify both stderr and tee_stderr'
        self.new_in_file = stdin
        self.new_out_file = stdout
        self.new_err_file = stderr
        self.tee_out_file = tee_stdout
        self.tee_err_file = tee_stderr
        self.append_stdout = append_stdout
        self.append_stderr = append_stderr

    def __enter__(self):
        out_mode = 'a' if self.append_stdout else 'w'
        err_mode = 'a' if self.append_stderr else 'w'
        if self.new_in_file is not None:
            self.new_in = open(self.new_in_file, 'r')
            self.old_in = sys.stdin
            sys.stdin = self.new_in
        if self.new_out_file is not None:
            self.new_out = open(self.new_out_file, out_mode)
            self.old_out = sys.stdout
            sys.stdout = self.new_out
        if self.new_err_file is not None:
            self.new_err = open(self.new_err_file, err_mode)
            self.old_err = sys.stderr
            sys.stderr = self.new_err
        if self.tee_out_file is not None:
            self.new_out = Tee(sys.stdout, open(self.tee_out_file, out_mode))
            self.old_out = sys.stdout
            sys.stdout = self.new_out
        if self.tee_err_file is not None:
            self.new_err = Tee(sys.stderr, open(self.tee_err_file, err_mode))
            self.old_err = sys.stderr
            sys.stderr = self.new_err

    def __exit__(self, exctype, excinst, exctb):
        if self.new_in_file is not None:
            sys.stdin = self.old_in
            self.new_in.close()
        if self.new_out_file is not None:
            sys.stdout = self.old_out
            self.new_out.close()
            consolidate_file(self.new_out_file, self.new_out_file)
        if self.new_err_file is not None:
            sys.stderr = self.old_err
            self.new_err.close()
            consolidate_file(self.new_err_file, self.new_err_file)
        if self.tee_out_file is not None:
            sys.stdout = self.old_out
            self.new_out.outputs[1].close()
            consolidate_file(self.tee_out_file, self.tee_out_file)
        if self.tee_err_file is not None:
            sys.stderr = self.old_err
            self.new_err.outputs[1].close()
            consolidate_file(self.tee_err_file, self.tee_err_file)


class Tee:
    def __init__(self, *outputs: TextIO):
        self.outputs  = outputs
    def write(self, *args, **kwargs):
        for out in self.outputs:
            out.write(*args, **kwargs)
    def flush(self, *args, **kwargs):
        for out in self.outputs:
            out.flush(*args, **kwargs)


@contextmanager
def maybe_open(filename: Optional[str], *args, default=None, **kwargs):
    if filename is not None:
        f = open(filename, *args, **kwargs)
        has_opened = True
    else:
        f = default
        if f is not None:
            f.flush()
        has_opened = False
    try:
        yield f
    except Exception:
        raise
    finally:
        if has_opened:
            f.close()


def consolidate_file(infile: Path, outfile: Path) -> None:
    '''Remove "erased lines" from a text file, 
    e.g. "Example:\nTo whom it may concern,\rHello,\rHi,\nthis is an example.\n" -> "Example:\nHi,\nthis is an example.\n" '''
    CR = ord(b'\r')
    LF = ord(b'\n')
    with open(infile, 'rb') as r:
        original = r.read().replace(b'\r\n', b'\n')
    current_line = []
    with open(outfile, 'wb') as w:
        for byte in original:
            if byte == LF:
                current_line.append(byte)
                w.write(bytes(current_line))
                current_line.clear()
            elif byte == CR:
                current_line.clear()
            else:
                current_line.append(byte)
        w.write(bytes(current_line))

def consolidate_string(original: str) -> str:
    '''Remove "erased lines" from a text file, 
    e.g. "Example:\nTo whom it may concern,\rHello,\rHi,\nthis is an example.\n" -> "Example:\nHi,\nthis is an example.\n" '''
    CR = '\r'
    LF = '\n'
    result = []
    current_line = []
    for char in original:
        if char == LF:
            current_line.append(char)
            result.extend(current_line)
            current_line.clear()
        elif char == CR:
            current_line.clear()
        else:
            current_line.append(char)
    result.extend(current_line)
    return ''.join(result)
