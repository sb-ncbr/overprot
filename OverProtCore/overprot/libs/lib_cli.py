from __future__ import annotations
from dataclasses import dataclass
import inspect
import argparse
from collections import defaultdict
from pathlib import Path  # used in evals
from typing import Callable, TypeVar, Generic, Type, Union, Optional, Literal, Any, get_origin, get_args
from typing import _GenericAlias  # type: ignore

_PRINT_ALL  = False

_Func = TypeVar('_Func')
_T = TypeVar('_T')

_EMPTY = inspect._empty  # type: ignore


class LibCliException(Exception): ...


class _CliParam(Generic[_T]):
    name: str
    inspect_type: Any
    type: Type|_GenericAlias 
    parser: Callable[[str], _T]|None
    default: _T
    is_option: bool  # Like --size 5, --reverse
    is_switch: bool  # Like --reverse
    cli_names: list[str]
    help: str

    def __init__(self, file: str, function_name: str, doc: _ParsedDocstring, param: inspect.Parameter, 
            short_options: dict[str, str], parsers: dict[str, Callable[[str], _T]]) -> None:
        self.name = param.name
        if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD:
            raise NotImplementedError(f'@cli_command not implemented for *args and **kwargs (function `{function_name}` in {file})')
        if param.annotation == _EMPTY:
            raise LibCliException(f'Missing type annotation for parameter `{param.name}` of function `{function_name}` in {file}')
        self.inspect_type = param.annotation
        try:
            self.type = _decode_type(self.inspect_type)
        except NotImplementedError:
            if self.name in parsers:
                self.type = None  # OK
            else:
                raise
        self.default = param.default
        self.is_option = self.default != _EMPTY
        self.is_switch = self.type == bool
        if self.is_switch:
            if self.default != False:
                raise LibCliException(f'Parameter of type bool must have default value False (parameter `{param.name}` of function `{function_name}` in {file})')
            self.parser = None
        else:
            self.parser = parsers.get(self.name) or _get_parser(self.type)
        if self.is_option:
            self.cli_names = [f'--{self.name}']
            if self.name in short_options:
                short_name = short_options[self.name]
                if short_name is not None and not (len(short_name) == 2 and short_name[0] == '-' and short_name[1].isalpha()):
                    raise LibCliException(f'Short options must be of format -X, where X is a letter (function `{function_name}` in {file})')
                self.cli_names.append(short_name)
        else:
            self.cli_names = [self.name]
        self.help = doc.params.get(self.name, '')
        if self.is_option and not self.is_switch and self.default is not None:
            self.help += f'\n[default: {self.default}]'

    def __str__(self) -> str:
        parts = [f'{k}={repr(v)}' for k, v in vars(self).items()]
        return f'PARAM ({", ".join(parts)})'
    
    def add_as_argument(self, parser: argparse.ArgumentParser) -> None:
        kwargs: dict[str, Any] = {'help': self.help}
        if self.is_switch:
            kwargs['action'] = 'store_true'
        elif self.is_option:
            assert self.parser is not None
            kwargs['type'] = self.parser
            kwargs['default'] = self.default
        else:
            assert self.parser is not None
            kwargs['type'] = self.parser
        if isinstance(self.type, _GenericAlias) and get_origin(self.type) == Literal:
            kwargs['choices'] = get_args(self.type)
        parser.add_argument(*self.cli_names, **kwargs)


class _CliInfo(object):
    name: str
    help: str|None
    params: list[_CliParam]

    def __init__(self, function: Callable, short_options: dict[str, str], parsers: dict[str, Callable[[str], _T]]) -> None:
        assert callable(function), '@cli_command can only be applied to a function'
        assert hasattr(function, '__name__'), '@cli_command can only be applied to a function with __name__'
        assert hasattr(function, '__code__'), '@cli_command can only be applied to a function with __code__'
        self.name = function.__name__
        file = function.__code__.co_filename
        doc = _ParsedDocstring(function.__doc__)
        self.help = doc.main
        signature = inspect.signature(function)
        self.params = [_CliParam(file, self.name, doc, p, short_options, parsers) for p in signature.parameters.values()]
        param_names = {p.name for p in self.params}
        for param in doc.params:
            if param not in param_names:
                raise LibCliException(f'Docstring for function `{self.name}` contains extra @param {param} ({file})')
        return_type = _decode_type(signature.return_annotation)
        if return_type not in (int, None, Optional[int]):
            raise LibCliException(f'Return type of `{self.name}` must be int or None or int|None ({file})')

    def __str__(self) -> str:
        lines = []
        lines.append(f'FUNCTION {self.name}')
        lines.append(f'    HELP {self.help}')
        for param in self.params:
            lines.append(f'    {param}')
        return '\n'.join(lines)

    def parse_args(self) -> dict[str, Any]:
        parser = argparse.ArgumentParser(description=self.help)
        for param in self.params:
            param.add_as_argument(parser)
        args = parser.parse_args()
        return vars(args)

    def log(self) -> None:
        '''For debugging'''
        print(self)
        print()


@dataclass
class _ParsedDocstring(object):
    main: str  # main text of the docstring
    params: dict[str, str]  # descriptions of parameters (found after @param )
    return_: str  # description of return value (found after @return)

    def __init__(self, docstring: str|None) -> None:
        docstring = docstring or ''
        lines = [line.strip() for line in docstring.strip().splitlines()]
        main_lines = []
        param_lines: dict[str, list[str]] = defaultdict(list)
        return_lines = []
        context: str|None = None  # None = parsing main context, xxx = parsing @param xxx, return = parsing @return
        for line in lines:
            if line == '':
                if context is None:
                    main_lines.append(line)
                context = None
            else:
                if line.startswith('@param'):
                    parts = line.split(maxsplit=2)
                    param = parts[1]
                    if param.startswith('`') and param.endswith('`'):
                        param = param[1:-1]
                    line = parts[2] if len(parts) > 2 else ''
                    context = param
                elif line.startswith('@return'):
                    parts = line.split(maxsplit=1)
                    line = parts[1] if len(parts) > 1 else ''
                    context = 'return'
                if context is None:
                    main_lines.append(line)
                elif context == 'return':
                    return_lines.append(line)
                else:
                    param_lines[context].append(line)
        self.main = '\n'.join(main_lines)
        self.params = {param: '\n'.join(lines) for param, lines in param_lines.items()}
        self.return_ = '\n'.join(return_lines)


class _CliCommandDecorator(object):
    '''Allow the decorated function to be run as a CLI command using run_cli_command(function)'''
    short_options: dict[str, str]
    parsers: dict[str, Callable[[str], _T]]
    def __init__(self, short_options: dict[str, str], parsers: dict[str, Callable[[str], _T]]) -> None:
        self.short_options = short_options
        self.parsers = parsers
    def __call__(self, function: _Func) -> _Func:
        cli_info = _CliInfo(function, self.short_options, self.parsers)  # type: ignore
        if _PRINT_ALL:
            cli_info.log()
        function._cli_info = cli_info  # type: ignore
        return function


def _decode_type(annotation: Any) -> Type|_GenericAlias:
    '''Convert type annotation from inspect.signature to a type'''
    # First evaluate `annotation` if it's string
    if isinstance(annotation, str):
        try:
            annotation = eval(annotation)
        except Exception:
            # Try parsing as X|None
            try: 
                types = [eval(p) for p in annotation.split('|')]
            except Exception:
                raise NotImplementedError(f'Cannot decode type {repr(annotation)}')
            if len(types) == 2 and None in types:
                base_type = types[0] if types[0] != None else types[1]
                return Optional[base_type]
            else:
                raise NotImplementedError(f'Not supported for Union types except Optional[X] ({repr(annotation)})')
    # Check if it's supported type
    if isinstance(annotation, type):
        return annotation
    elif isinstance(annotation, _GenericAlias) and get_origin(annotation) == Union:
        if len(get_args(annotation)) == 2 and type(None) in get_args(annotation):
            return annotation
        else:
            raise NotImplementedError(f'Not supported for Union types except Optional[X] ({repr(annotation)})')
    elif isinstance(annotation, _GenericAlias) and get_origin(annotation) == Literal:
        if all(isinstance(value, str) for value in get_args(annotation)):
            return annotation
        else:
            raise LibCliException(f'Literal types are only suppored, if all values are strings ({repr(annotation)})')
    elif annotation is None:
        return annotation
    else:
        raise NotImplementedError(f'_decode_type({repr(annotation)})')

def _get_parser(typ: Any) -> Callable[[str], Any]:
    '''Convert a type (including Optional[T]) to a function that parses a string to this type'''
    if typ == bool:
        raise AssertionError()
    elif isinstance(typ, type):
        return typ
    elif isinstance(typ, _GenericAlias) and get_origin(typ) == Union:
        types = get_args(typ)
        if len(types) == 2 and type(None) in types:
            base_type = types[0] if types[0] != type(None) else types[1]
        return base_type
    elif isinstance(typ, _GenericAlias) and get_origin(typ) == Literal:
        return str
    else:
        raise NotImplementedError(f'_get_parser({repr(typ)})')


def cli_command(short_options: dict[str, str] = {}, parsers: dict[str, Callable[[str], _T]] = {}) -> _CliCommandDecorator:
    """Allow the decorated function to be run as a CLI command using `run_cli_command(function)`.
    
    Example:
    
    ```
    @cli_command(short_options={'output': '-o', 'verbose': '-v'}, parsers={'number': lambda s: 0 if s == 'zero' else int(s)})
    def foo(number: int, output: None | Path = None, verbose: bool = False) -> int:
        '''Example function `foo`.
        @param  `number`   Number to be processed.
        @param  `output`   Name of output file (default: stdout).
        @param  `verbose`  Print more stuff.
        @return  0 if successful, 1 otherwise.
        This function can also be run from command line, e.g.:
        python foo.py 5 --verbose -o example.txt
        '''
        ...
    if __name__ == '__main__:
        run_cli_command(foo)
    ```
    """
    if not isinstance(short_options, dict) or not isinstance(parsers, dict):
        raise LibCliException("Parameters of @cli_command() must be dicts, didn't you forget ()?")
    return _CliCommandDecorator(short_options, parsers)

def run_cli_command(function: Callable) -> None:
    '''Run `function` as a CLI command (i.e. read the command line arguments and pass them to the function). The exit code will be the return value of the function (or 0 if None).'''
    try:
        cli_info: _CliInfo = function._cli_info  # type: ignore
    except AttributeError:
        try:
            name = function.__name__
        except AttributeError:
            name = ''
        raise LibCliException(f'Function {name} must be decorated with @cli_command if it is to be run with run_cli_command')
    args = cli_info.parse_args()
    exit_code = function(**args)
    if exit_code is not None:
        exit(exit_code)
