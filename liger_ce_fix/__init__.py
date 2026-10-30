"""Liger fused linear cross-entropy with the log-space softmax-collapse fix (arXiv 2610.31642)."""
from liger_ce_fix._meta import commit as _commit, version as _version

__version__ = _version()
__commit__ = _commit()


def cross_entropy(*args, **kwargs):  # replaced in Task 5
    raise NotImplementedError("cross_entropy lands in Task 5")


__all__ = ["cross_entropy", "__version__", "__commit__"]
