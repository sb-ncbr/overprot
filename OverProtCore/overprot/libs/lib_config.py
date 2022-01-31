'''
Reading structured configuration from .ini files.
Configurations are specified by subclassing Config.
'''

import configparser
from pathlib import Path
from typing import List,  Dict, Optional, Literal, Final, Type, Union, get_origin, get_args, get_type_hints


_ConfigOptionValue = Union[str, int, float, bool, Path, List[str], List[int], List[float], List[bool], List[Path]]


class ConfigException(Exception):
    pass


class ConfigSection(object):
    '''Represents one section of configuration like in .ini file.
    Subclasses of ConfigSection can declare instance variables corresponding to individual options in the section, 
    these should be of one of the types given by _ConfigOptionValue.
    Instance variables with prefix _ are ignored.
    '''
    __ALLOWED_TYPES: Final = get_args(_ConfigOptionValue)
    __DEFAULT_TYPE: Final = str
    __option_types: Dict[str, type]

    def __init__(self):
        cls = type(self)
        self.__option_types = {option: typ for option, typ in get_type_hints(cls).items() if not option.startswith('_')}
        for option, typ in self.__option_types.items():
            allowed_types = ', '.join(str(t) for t in self.__ALLOWED_TYPES) + ' or Literal of strings'
            assert typ in self.__ALLOWED_TYPES or self._is_string_literal(typ), f'Type {typ} is not allowed for options in Config ({cls.__name__}.{option}). Allowed types: {allowed_types}'
            if option in vars(cls):
                value = vars(cls)[option]
            elif get_origin(typ) is None:  # non-generic
                value = typ()
            elif get_origin(typ) is list:
                value = list()
            elif get_origin(typ) is Literal:
                value = get_args(typ)[0]
            else:
                raise NotImplementedError(typ)
            self.__setattr__(option, value)
    
    def __str__(self) -> str:
        lines = []
        for option in vars(self):
            if not option.startswith('_'):
                value = self.__getattribute__(option)
                if isinstance(value, str):
                    value = value.replace('\n', '\n\t')
                elif isinstance(value, list):
                    value = '\n\t'.join(str(elem) for elem in value)
                lines.append(f'{option} = {value}')
        return '\n'.join(lines)

    def __repr__(self) -> str:
        opts = []
        for option in vars(self):
            if not option.startswith('_'):
                value = self.__getattribute__(option)
                opts.append(f'{option}={repr(value)}')
        options = ', '.join(opts)
        return f'{type(self).__name__}({options})'

    def _set_options(self, parser: configparser.ConfigParser, section: str, allow_extra: bool = False, allow_missing: bool = False, filename: str = '???') -> None:
        options = parser[section]

        if not allow_extra:
            for option in options:
                if option not in self.__option_types:
                    raise ConfigException(f'Extra option "{option}" in section [{section}] in file {filename}')
        else:
            for option in options:
                if option not in self.__option_types:
                    self.__option_types[option] = self.__DEFAULT_TYPE
                    self.__setattr__(option, self.__DEFAULT_TYPE())

        if not allow_missing:
            for option in self.__option_types:
                if option not in options:
                    raise ConfigException(f'Missing option "{option}" in section [{section}] in file {filename}')
            
        for option in options:
            option_type = self.__option_types.get(option, self.__DEFAULT_TYPE)
            typed_value: _ConfigOptionValue
            if option_type == str:
                typed_value = parser.get(section, option)
            elif option_type == int:
                try:
                    typed_value = parser.getint(section, option)
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be an integer.')
            elif option_type == float:
                try:
                    typed_value = parser.getfloat(section, option)
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be a float.')
            elif option_type == bool:
                try:
                    typed_value = parser.getboolean(section, option)
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be a boolean (True/False).')
            elif option_type == Path:
                try:
                    typed_value = Path(parser.get(section, option))
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be a path.')
            elif get_origin(option_type) == list:
                item_type, = get_args(option_type)
                lines = parser.get(section, option).split('\n')
                converter = self._parse_bool if item_type == bool else item_type
                try:
                    typed_value = [converter(line) for line in lines if line != '']
                except ValueError:
                    value = parser.get(section, option)
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {value}. Must be a list of {option_type}, one per line.')
            elif get_origin(option_type) == Literal:
                allowed_values = get_args(option_type)
                typed_value = parser.get(section, option)
                if typed_value not in allowed_values:
                    raise ValueError(f'Option {option} in section [{section}] in file {filename} has invalid value {typed_value}. Must be one of: {", ".join(allowed_values)}.')
            else:
                raise NotImplementedError(f'Option type: {option_type}')
            self.__setattr__(option, typed_value)
    
    @staticmethod
    def _parse_bool(string: str) -> bool:
        low_string = string.lower()
        if low_string in ('true', '1'):
            return True
        elif low_string in ('false', '0'):
            return False
        else:
            raise ValueError(f'Not a boolean: {string}')
    
    @staticmethod
    def _is_string_literal(typ: Type) -> bool:
        return get_origin(typ) == Literal and all(isinstance(v, str) for v in get_args(typ))


class Config(object):
    '''Represents configuration like in .ini file.
    Subclasses of Config can declare instance variables corresponding to individual configuration sections, these should be subclasses of ConfigSection.
    Instance variables with prefix _ are ignored.
    '''
    __SECTION_TYPE: Final = ConfigSection
    __section_types: Dict[str, Type[ConfigSection]]

    def __init__(self, filename: Optional[Path] = None, allow_extra: bool = False, allow_missing: bool = False):
        '''Create new Config object with either default values or loaded from an .ini file.'''
        cls = type(self)
        self.__section_types = {section: typ for section, typ in get_type_hints(cls).items() if not section.startswith(f'_')}
        for section, typ in self.__section_types.items():
            assert issubclass(typ, self.__SECTION_TYPE)
            value = vars(cls).get(section, typ())
            self.__setattr__(section, value)
        if filename is not None:
            self.load_from_file(filename, allow_extra=allow_extra, allow_missing=allow_missing)

    def __str__(self) -> str:
        lines = []
        for section in vars(self):
            if not section.startswith('_'):
                value = self.__getattribute__(section)
                lines.append(f'[{section}]')
                lines.append(str(value))
                lines.append('')
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        sects = []
        for section in vars(self):
            if not section.startswith('_'):
                value = self.__getattribute__(section)
                sects.append(f'{section}={repr(value)}')
        return f'{type(self).__name__}({", ".join(sects)})'
    

    def load_from_file(self, filename: Path, allow_extra: bool = False, allow_missing: bool = False) -> None:
        '''Load configuration options from an .ini file into this Config object.'''
        parser = configparser.ConfigParser()
        with open(filename) as r:
            parser.read_file(r)
        loaded_sections = parser.sections()

        if not allow_extra:
            for section in loaded_sections:
                if section not in self.__section_types:
                    raise ConfigException(f'Extra section [{section}] in {filename}')
        else:
            for section in loaded_sections:
                if section not in self.__section_types:
                    self.__section_types[section] = self.__SECTION_TYPE
                    self.__setattr__(section, self.__SECTION_TYPE())

        if not allow_missing:
            for section in self.__section_types:
                if section not in loaded_sections:
                    raise ConfigException(f'Missing section [{section}] in {filename}')
            
        for section in loaded_sections:
            self.__getattribute__(section)._set_options(parser, section, allow_extra=allow_extra, allow_missing=allow_missing, filename=filename)
