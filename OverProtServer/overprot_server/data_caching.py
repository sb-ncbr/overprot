from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generic, TypeVar, Iterable, Callable, Optional


DEFAULT_VALIDITY = timedelta(hours=12)
DEFAULT_VALIDITY_WITH_WATCHFILES = timedelta(days=30)
DEFAULT_THROTTLE_WITH_WATCHFILES = timedelta(minutes=1)

V = TypeVar('V')


class DataCache(Generic[V]):
    '''Recomputes the value if the current value is older than `validity`.'''
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

    def _recompute_needed(self) -> bool:
        return self._valid_until is None or datetime.now() > self._valid_until

    @property
    def value(self) -> V:
        if self._recompute_needed():
            self._value = self._factory()  # type: ignore
            self._valid_until = datetime.now() + self._validity
        return self._value  # type: ignore  # (_value has been set, but cannot assert _value is not None, because None might be a valid value for V)
 

class DataCacheWithWatchfiles(DataCache[V]):
    '''Recomputes the value if the current value is older than `validity` or if any of the `watchfiles` has changed.
    However, if the last value request was more recently than `throttle_time`, does not recompute.'''
    _watchfile_mtimes: dict[Path, float]
    _throttle_time: timedelta
    _throttle_until: Optional[datetime]
    
    def __init__(self, factory: Callable[[], V], watchfiles: Iterable[Path], validity: timedelta = DEFAULT_VALIDITY_WITH_WATCHFILES, throttle_time: timedelta = DEFAULT_THROTTLE_WITH_WATCHFILES) -> None:
        super().__init__(factory, validity)
        self._watchfile_mtimes = {file: float('nan') for file in watchfiles}
        self._throttle_time = throttle_time
        self._throttle_until = None
    
    def _recompute_needed(self) -> bool:
        if self._do_throttle():
            return False
        changed = False
        for file in self._watchfile_mtimes:
            mtime = file.stat().st_mtime
            if mtime != self._watchfile_mtimes[file]:
                changed = True
                self._watchfile_mtimes[file] = mtime
                # Do not break, update mtimes of all files
        return changed or super()._recompute_needed()
    
    def _do_throttle(self) -> bool:
        now = datetime.now()
        if self._throttle_until is None or now > self._throttle_until:
            self._throttle_until = now + self._throttle_time
            return False
        else:
            return True