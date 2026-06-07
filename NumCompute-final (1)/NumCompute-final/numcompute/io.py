"""Input/output helpers for NumCompute."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def load_csv(
    file_path: str,
    delimiter: str = ",",
    missing_value: float | int | str = np.nan,
    skip_header: int = 0,
    dtype: Any = float,
    encoding: str | None = None,
) -> np.ndarray:
    """
    Load a delimited text file into a NumPy array.

    Parameters
    ----------
    file_path : str
        Path to the input file.
    delimiter : str, default=","
        Field delimiter.
    missing_value : scalar, default=np.nan
        Replacement value for missing entries.
    skip_header : int, default=0
        Number of initial rows to skip.
    dtype : data-type, default=float
        Desired dtype of the returned array.
    encoding : str or None, default=None
        Optional file encoding passed to NumPy.

    Returns
    -------
    np.ndarray
        Loaded data as a NumPy array.

    Raises
    ------
    TypeError
        If argument types are invalid.
    ValueError
        If the file cannot be loaded or parsed.

    Notes
    -----
    Time complexity is O(n) in the number of entries read.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")
    if not isinstance(delimiter, str) or len(delimiter) == 0:
        raise TypeError("delimiter must be a non-empty string.")
    if not isinstance(skip_header, int) or skip_header < 0:
        raise ValueError("skip_header must be a non-negative integer.")
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    kwargs = {
        "delimiter": delimiter,
        "skip_header": skip_header,
        "filling_values": missing_value,
        "dtype": dtype,
    }
    if encoding is not None:
        kwargs["encoding"] = encoding

    try:
        data = np.genfromtxt(file_path, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Error loading CSV: {exc}") from exc

    return np.asarray(data)
