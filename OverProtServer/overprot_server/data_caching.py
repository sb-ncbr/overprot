from datetime import datetime, timedelta
from typing import Generic, TypeVar, Callable, Optional


DEFAULT_VALIDITY = timedelta(hours=1)
V = TypeVar('V')


class DataCache(Generic[V]):
    _factory: Callable[[], V]
    _value: Optional[V]
    _validity: timedelta
    _valid_until: Optional[datetime]

    def __init__(self, factory: Callable[[], V], validity: timedelta = DEFAULT_VALIDITY) -> None:
        super().__init__()
        self._factory = factory
        self._value = None
        self._validity = validity
        self._valid_until = None

    @property
    def value(self) -> V:
        if self._valid_until is None or datetime.now() > self._valid_until:
            self._value = self._factory()
            self._valid_until = datetime.now() + self._validity
            # raise Exception(f'NuNuNu {self._factory}')
        return self._value
