'''
Building pipelines with `Pipeline` and running parts of the pipelines.
'''

from __future__ import annotations
from typing import Callable  #, TypeVar, ParamSpec, Concatenate


class PipelineStepSpecificationError(Exception):
    '''Raised when the specification of steps to run is invalid, e.g. refers to a non-existing step name.'''


class PipelineDefinitionError(Exception):
    '''Raised when a `Pipeline` is ill-defined, e.g. two steps have the same name.'''


class Step(object):
    _name: str
    _body: Callable[[], None]
    _always: bool

    def __init__(self, body: Callable[[], None], always: bool = False) -> None:
        self._name = body.__name__
        self._body = body
        self._always = always
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}[{self._name}]'

    def __call__(self) -> None:
        self._body()


class Pipeline(object):
    '''This class allows defining a pipeline (i.e. a sequence of steps) 
    and specifying which particular steps should be run.

    Example:
    ```
    with Pipeline('EXAMPLE_PIPELINE', steps='A:B+') as pipeline:
        @pipeline.step
        def A():
            ...
        @pipeline.always
        def B():
            ...
        @pipeline.step
        def C():
            ...
    ```

    `steps` defines which steps should be run. 
    It is a comma-separated list of items, where each item is either a step name, or a range of steps, e.g. 'A:B+'.
    The + symbol after a step name instructs to start/end after that step, without + it starts/ends before that step.
    E.g. `steps=':D, G+:K, P:R+'` will run steps A, B, C, H, I, J, P, Q, R.
    Steps defined with `.always` (instead of `.step`) will be run even if not listed.
    `steps='?'` will print the list of all steps, without actually running any steps.
    '''
    _name: str
    _help: str
    _step_list: list[Step]
    _step_index: dict[str, int]  # {name: index}
    _steps_to_run: str
    _print_step_headers: bool
    _step_spec_error_handler: Callable[[PipelineStepSpecificationError], None] | None

    def __init__(self, name: str, *, help: str = '', steps: str = ':', print_step_headers: bool = True, 
            step_spec_error_handler: Callable[[PipelineStepSpecificationError], None] | None = None) -> None:
        self._name = name
        self._help = help
        self._step_list = []
        self._step_index = {}
        self._steps_to_run = steps.replace(' ', '')
        self._print_step_headers = print_step_headers
        self._step_spec_error_handler = step_spec_error_handler
    
    def __enter__(self) -> Pipeline:
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_type is None:
            if self._steps_to_run == '?':
                if self._help != '':
                    print(self._help)
                print(self.list_steps(header=True))
            else:
                try:
                    selected_steps = self._parse_step_specs(self._steps_to_run)
                except PipelineStepSpecificationError as err:
                    selected_steps = []
                    if self._step_spec_error_handler is not None:
                        self._step_spec_error_handler(err)
                        return
                    else:
                        raise
                for i_step, step in enumerate(self._step_list):
                    do_run = step._always or i_step in selected_steps
                    if do_run:
                        if self._print_step_headers:
                            print(f'::: {step._name} :::')
                        try:
                            step.__call__()
                        finally:
                            if self._print_step_headers:
                                print()
    
    def __repr__(self) -> str:
       return  f'{self.__class__.__name__}[{self._name}, {len(self._step_list)}]'
    
    def _parse_step_specs(self, specs: str) -> set[int]:
        selected_steps: set[int] = set()
        for spec in specs.split(','):
            spec_type = spec.count(':')
            if spec_type == 0:
                the_step = self._step_name_to_index(spec, specs)
                if the_step is not None:
                    selected_steps.add(the_step)
            elif spec_type == 1:
                fro, to = spec.split(':')
                start_step = self._step_name_to_index(fro, specs) or 0
                end_step = self._step_name_to_index(to, specs) or len(self._step_list)
                selected_steps.update(range(start_step, end_step))
            else:
                raise PipelineStepSpecificationError(f"Invalid range '{spec}' in step specification '{specs}' (too many colons)")
        return selected_steps
    
    def _step_name_to_index(self, name: str, full_specs: str|None = None) -> int|None:
        if name == '':
            return None
        else:
            shift = 1 if name.endswith('+') else 0
            if shift: 
                name = name[:-1]
            i = self._step_index.get(name)
            if i is None:
                msg = f"Unknown step name '{name}'"
                if full_specs is not None:
                    msg += f" in step specification '{full_specs}'"
                raise PipelineStepSpecificationError(msg)
            if shift:
                i += 1
            return i
        
    def _add_step(self, func: Callable[[], None], always: bool) -> Step:
        new_step = Step(func, always=always)
        if new_step._name in self._step_index:
            raise PipelineDefinitionError(f"Duplicate step name: '{new_step._name}'")
        self._step_list.append(new_step)
        self._step_index[new_step._name] = len(self._step_index)
        return new_step
    
    def step(self, func: Callable[[], None]) -> Step:
        '''Add a new step to the pipeline. 
        Usage:
        ```
        @pipeline.step
        def STEP_XY():
            ...
        ```
        '''
        return self._add_step(func, always=False)
    
    def always(self, func: Callable[[], None]) -> Step:
        '''Add a new obligatory step to the pipeline. 
        (i.e. this step will be performed always, even if not specified by `steps`)
        Usage:
        ```
        @pipeline.always
        def STEP_XY():
            ...
        ```
        '''
        return self._add_step(func, always=True)

    def list_steps(self, header: bool = False) -> str:
        '''Return a formatted list of all steps in this pipeline.
        Add '(always)' to obligatory steps.
        '''
        lines = []
        if header:
            line = f'Pipeline {self._name} steps:'
            lines.append(line)
        width = max(len(step._name) for step in self._step_list)
        for step in self._step_list:
            flag = '(always)  ' if step._always else ''
            doc = step._body.__doc__ or ''
            line = f'  - {step._name:{width}}  {flag}{doc}'
            lines.append(line)
        return '\n'.join(lines)

    
    
# P1 = ParamSpec('P1')
# R1 = TypeVar('R1')
# P2 = ParamSpec('P2')
# R2 = TypeVar('R2')
# DP = ParamSpec('DP')
# T = TypeVar('T')

# def parametrized_decorator(d: Callable[Concatenate[Callable[P1, R1], DP], Callable[P2, R2]]) -> Callable[DP, Callable[[Callable[P1, R1]], Callable[P2, R2]]]:
#     '''Allow definition of decorators with parameters like this:
#         ```
#         @parametrized_decorator
#         def decorator(function: Callable[P1, R1], *args, *kwargs) -> Callable[P2, R2]:
#             ...
#         @decorator(1, 2, 3, name='blabla')
#         def foo():
#             ...
#         ```
#     '''
#     def decorator_factory(*args, **kwargs) -> Callable[[Callable[P1, R1]], Callable[P2, R2]]:
#         def final_decorator(function: Callable[P1, R1]) -> Callable[P2, R2]:
#             return d(function, *args, **kwargs)
#         return final_decorator
#     return decorator_factory


if __name__ == '__main__':
    with Pipeline('EXAMPLE_PIPELINE', steps='?', help='This is example help message.') as pipeline:

        @pipeline.step
        def A():
            print('a')

        @pipeline.always
        def B():
            print('b')

        @pipeline.step
        def C():
            print('c')

