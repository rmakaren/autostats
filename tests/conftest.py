import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def missing_df(num_rows, num_cols, num_missing):
    data = np.random.uniform(low=1, high=100, size=(num_rows, num_cols))
    mask = np.random.choice([True, False], size=data.shape, p=[num_missing/(num_rows*num_cols), 1-num_missing/(num_rows*num_cols)])
    data[mask] = np.nan
    df = pd.DataFrame(data)
    return df