import numpy as np
import tempfile

from numcompute.io import load_csv
import pytest

def test_invalid_path_raises():
    with pytest.raises(ValueError):
        load_csv("non_existent.csv")

def test_tab_delimiter():
    data = "1\t2\n3\t4"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.flush()
        arr = load_csv(f.name, delimiter="\t")
    assert arr.shape == (2, 2)

def test_invalid_skip_header():
    import pytest
    with pytest.raises(ValueError):
        load_csv("file.csv", skip_header=-1)

def test_invalid_file_path_type():
    import pytest
    with pytest.raises(TypeError):
        load_csv(123)            

def test_load_csv_basic():
    data = "1,2\n3,4"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.flush()
        arr = load_csv(f.name)
    assert arr.shape == (2, 2)
    assert np.array_equal(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_load_csv_missing_values():
    data = "1,\n3,4"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.flush()
        arr = load_csv(f.name)
    assert np.isnan(arr[0, 1])


def test_load_csv_skip_header():
    data = "a,b\n1,2\n3,4"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.flush()
        arr = load_csv(f.name, skip_header=1)
    assert arr.shape == (2, 2)
