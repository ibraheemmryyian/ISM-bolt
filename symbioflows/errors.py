class SymbioFlowsException(Exception):
    """Standardised, typed error for all SymbioFlows services."""
    def __init__(self, code: str, http_status: int, message: str, *, meta: dict | None = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.meta = meta or {}