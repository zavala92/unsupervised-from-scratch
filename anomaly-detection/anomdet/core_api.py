"""Thin re-export layer so stages and tests can say `from anomdet import ...`
regardless of whether core.py or reference_impl.py is active."""

from . import core as _c

__all__ = [
    "fit_independent",
    "independent_pdf",
    "log_independent_pdf",
    "precision_recall_f1",
    "tune_threshold",
    "tune_threshold_log",
    "fit_correlated",
    "log_correlated_pdf",
    "correlated_pdf",
]


def __getattr__(name):
    if name in __all__:
        return getattr(_c, name)
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
