import pytest
import pandas as pd
import numpy as np
import random

from typing import Callable, List, Optional, Tuple, Union


@pytest.fixture
def generate_df_NaN() -> Callable[..., str]:
    def _generate_df_NaN(num_rows:int, num_cols:int, num_missing:int) -> pd.DataFrame:
        """_summary_

        Args:
            num_rows (int): _description_
            num_cols (int): _description_
            num_missing (int): _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = np.random.uniform(low=1, high=100, size=(num_rows, num_cols))
        mask = np.random.choice([True, False], size=data.shape, p=[num_missing/(num_rows*num_cols), 1-num_missing/(num_rows*num_cols)])
        data[mask] = np.nan
        df = pd.DataFrame(data)
        df["labels"] = random.choices(["label1", "label2"], k=num_rows)
        return df
    return _generate_df_NaN