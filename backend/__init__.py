
"""Backend package root.

In many historical modules we used the shorthand import style::

    from utils.advanced_data_validator import AdvancedDataValidator

Those modules expect *utils* to be importable from the global namespace.
Rather than refactor every call-site we make *backend.utils* available under
that alias at import-time.  This keeps legacy code working while preserving a
clear package hierarchy.
"""

import sys
from importlib import import_module

# Register alias only once – if another sub-module already imported the alias
# we leave the existing reference intact (important during reloads / tests).
if 'utils' not in sys.modules:
    sys.modules['utils'] = import_module('backend.utils') 