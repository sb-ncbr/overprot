from typing import NamedTuple, Optional, List


DOMAIN_FIELD_SEPARATOR = ','
RANGE_SEPARATOR = ':'
DEFAULT_CHAIN = 'A'

COMMENT_SYMBOL = '#'


class DomainParsingError(Exception):
    pass


class Range(NamedTuple):
    start: Optional[int]
    end: Optional[int]

    @classmethod
    def parse(cls, string: str) -> 'Range':
        start_end = string.split(RANGE_SEPARATOR)
        if len(start_end) != 2:
            raise DomainParsingError(f'Invalid range: {string}')
        s_start, s_end = start_end
        s_start = s_start.strip()
        s_end = s_end.strip()
        try:
            start = int(s_start) if s_start != '' else None
            end = int(s_end) if s_end != '' else None
            return cls(start, end)
        except ValueError:
            raise DomainParsingError(f'Invalid range: {string}')

    def __repr__(self) -> str:
        s = str(self.start) if self.start is not None else ''
        e = str(self.end) if self.end is not None else ''
        return f'{s}{RANGE_SEPARATOR}{e}'


class Domain(object):
    pdb: str
    chain: str
    ranges: List[Range]

    def __init__(self, pdb: str, chain: Optional[str] = None, ranges: Optional[List[Range]] = None) -> None:
        self.pdb = pdb
        self.chain = chain if chain is not None else DEFAULT_CHAIN
        self.ranges = ranges if ranges is not None else [Range(None, None)]

    @classmethod
    def parse(cls, string: str) -> 'Domain':
        fields = string.split(DOMAIN_FIELD_SEPARATOR)
        pdb = fields[0].strip()
        if not _is_valid_pdb(pdb):
            raise DomainParsingError(f'{string} is not a valid PDB identifier')
        pdb = pdb.lower()
        if not pdb.isalnum():
            raise DomainParsingError(string)
        chain: Optional[str]
        if len(fields) >= 2:
            chain = fields[1].strip()
            if chain == '':
                raise DomainParsingError('Chain must not be empty')
        else:
            chain = None
        ranges: Optional[List[Range]]
        if len(fields) >= 3:
            try:
                ranges = [Range.parse(rang) for rang in fields[2:]]
            except DomainParsingError:
                raise DomainParsingError(string)
        else:
            ranges = None
        return cls(pdb, chain, ranges)

    def __repr__(self) -> str:
        ranges = DOMAIN_FIELD_SEPARATOR.join(str(r) for r in self.ranges)
        return f'{self.pdb}{DOMAIN_FIELD_SEPARATOR}{self.chain}{DOMAIN_FIELD_SEPARATOR}{ranges}'

    def __lt__(self, other: 'Domain') -> str:
        return repr(self) < repr(other)


def _is_valid_pdb(string: str) -> bool:
    return len(string) == 4 and string.isascii() and string.isalnum() and string[0].isdigit()


def parse_submission_list(lst: str) -> List[Domain]:
    lines = [line.split(COMMENT_SYMBOL, maxsplit=1)[0].strip() for line in lst.splitlines()]
    lines = [line for line in lines if line != '']
    domains: List[Domain] = []
    for i, line in enumerate(lines, start=1):
        try:
            domain = Domain.parse(line)
            domains.append(domain)
        except DomainParsingError as ex:
            raise DomainParsingError(f'Error in domain #{i}: {line} ({ex})')
    return domains
    