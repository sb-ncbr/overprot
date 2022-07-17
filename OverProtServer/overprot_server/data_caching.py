from datetime import datetime, timedelta
from typing import Generic, TypeVar, Callable, Optional


DEFAULT_VALIDITY = timedelta(hours=12)
V = TypeVar('V')


class DataCache(Generic[V]):
    _factory: Callable[[], V]
    _value: Optional[V]
    _validity: timedelta
    _valid_until: Optional[datetime]

    def __init__(self, factory: Callable[[], V], validity: timedelta = DEFAULT_VALIDITY) -> None:
        super().__init__()
        self._factory = factory  # type: ignore
        self._value = None
        self._validity = validity
        self._valid_until = None

    @property
    def value(self) -> V:
        if self._valid_until is None or datetime.now() > self._valid_until:
            self._value = self._factory()  # type: ignore
            self._valid_until = datetime.now() + self._validity
        return self._value  # type: ignore  # (_value has been set, but cannot assert _value is not None, because None might be a valid value for V)
