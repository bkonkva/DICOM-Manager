class ArgErrorType(Exception):
    """Raised when provided argument is of unsupported type."""
    pass

class UnreadableFileError(Exception):
    """Raised when pydicom cannot read provided file."""
    pass