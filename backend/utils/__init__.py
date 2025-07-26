
"""Utility sub-package for shared helper components.

Making *backend.utils* an explicit Python package ensures that imports such as
```
from utils.distributed_logger import DistributedLogger
```
and
```
from utils.advanced_data_validator import AdvancedDataValidator
```
resolve correctly whenever the *backend* directory is added to ``sys.path`` by
runtime entry-points (e.g. the Flask AI gateway).
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

# Eagerly re-export the two primary utility classes so that callers can rely on
# a concise import path while still getting static-analysis / IDE support.

AdvancedDataValidator: ModuleType = import_module("backend.utils.advanced_data_validator")  # type: ignore[assignment]
DistributedLogger: ModuleType = import_module("backend.utils.distributed_logger")  # type: ignore[assignment]

# Alias the classes at package level for convenience
if TYPE_CHECKING:  # pragma: no cover – hinting only
    from .advanced_data_validator import AdvancedDataValidator as _AdvancedDataValidator
    from .distributed_logger import DistributedLogger as _DistributedLogger

    __all__ = [  # noqa: WPS410 (necessary for re-export clarity)
        "AdvancedDataValidator",
        "DistributedLogger",
    ]

# At runtime, we need to expose the *actual* classes – not the modules – under
# the package namespace so that ``utils.AdvancedDataValidator`` works.
for _module, _name in [
    (AdvancedDataValidator, "AdvancedDataValidator"),
    (DistributedLogger, "DistributedLogger"),
]:
    globals()[_name] = getattr(_module, _name)