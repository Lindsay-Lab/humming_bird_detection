import inspect
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

types_int = ["id", "index"]
types_float = []
types_bool = []

dtypes = (
    dict.fromkeys(types_int, int)
    | dict.fromkeys(types_float, float)
    | dict.fromkeys(types_bool, bool)
)


def load_df(path, dtypes=dtypes):
    """Load a CSV into a DataFrame, coercing known column types and dropping unnamed index columns."""
    df = pd.read_csv(path, dtype=dtypes)
    df.drop(
        df.columns[df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    return df


_true_set = {"yes", "true", "t", "y", "1"}
_false_set = {"no", "false", "f", "n", "0"}


def str_to_bool(value, raise_exc=False):
    """Convert a string ('yes'/'no', 'true'/'false', '1'/'0', etc.) to a Python bool.

    Returns None (or raises ValueError if raise_exc=True) for unrecognised values.
    """
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False
    elif isinstance(value, bool):
        return value
    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def alias_kwargs(**aliases):
    """Decorator that maps deprecated keyword argument names to their canonical equivalents.

    Example::

        @alias_kwargs(centre_x='center_x')
        def fn(center_x): ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for alias, original in aliases.items():
                if alias in kwargs and original not in kwargs:
                    kwargs[original] = kwargs.pop(alias)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def filter_kwargs(function, kwargs):
    """Return a copy of kwargs containing only keyword arguments accepted by function."""
    args = inspect.signature(function).parameters.keys()
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in args}
    return filtered_kwargs
