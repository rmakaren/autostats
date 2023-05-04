
import os
import numpy as np
import pandas as pd
from pandas import testing
import pytest

from typing import Callable, List, Optional, Tuple, Union

from autostats.autostats import AutoStat
from sklearn.datasets import load_iris



class TestAutoTest:

    # we do remove NaN variables
    def test_preprocessing(self, generate_df_NaN:Callable, generate_df_duplicates:Callable):
        stat = AutoStat()

        # 1. common case
        dataset_NaN = generate_df_NaN(num_rows=100, num_cols=10, num_missing=10)
        df_preproc = stat.preprocessing(dataset = dataset_NaN,
                                        labels = "labels")

        assert dataset_NaN.isnull().sum().sum() > 0
        assert df_preproc.isnull().sum().sum() == 0
        assert type(df_preproc) == pd.DataFrame

        # 2. if NaN in all rows, returns None
        dataset_NaN_all = generate_df_NaN(num_rows=100, num_cols=10, num_missing=100*10)
        assert stat.preprocessing(dataset = dataset_NaN_all, labels = "labels") == None

        # 3. no NaN, should do nothing
        dataset_no_NaN = generate_df_NaN(num_rows=100, num_cols=10, num_missing=0)
        df_preproc3 = stat.preprocessing(dataset = dataset_no_NaN,
                                        labels = "labels") 

        testing.assert_frame_equal(dataset_no_NaN, df_preproc3)
        assert dataset_no_NaN.isnull().sum().sum() == 0
        assert df_preproc3.isnull().sum().sum() == 0

        # 4. no duplicates, should do nothing
        dataset_no_dupl = generate_df_duplicates(num_rows=100, num_cols=10, num_duplicates=0)
        df_preproc4 = stat.preprocessing(dataset = dataset_no_dupl,
                                        labels = "labels")
        
        testing.assert_frame_equal(dataset_no_dupl, df_preproc4)

        # 5. 1 duplicate, should remove duplicate
        dataset_dupl = generate_df_duplicates(num_rows=100, num_cols=10, num_duplicates=1)
        df_preproc5 = stat.preprocessing(dataset = dataset_dupl,
                                        labels = "labels")
        
        assert len(dataset_dupl) == len(df_preproc5) + 1
