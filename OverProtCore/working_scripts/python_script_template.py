'''
This Python script does foo ...

Example usage:
    python3  -m foo  --foo 4  foo.txt 
'''
# TODO add description and example usage in docstring

from __future__ import annotations
from overprot.libs.lib_cli import cli_command, run_cli_command


#  CONSTANTS  ################################################################################


#  FUNCTIONS  ################################################################################


#  MAIN  #####################################################################################

@cli_command()
def main(foo: str, bar: int = 123, baz: bool = False) -> int|None:
    # TODO add parameters
    '''Foo
    @param  foo  First parameter
    @param  bar  Second parameter
    @param  baz  Third parameter
    '''
    # TODO add docstring
    pass
    # TODO add implementation


if __name__ == '__main__':
    run_cli_command(main)
