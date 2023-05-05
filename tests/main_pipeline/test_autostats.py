
import os
import numpy as np
import pandas as pd
from pandas import testing
import pytest

from typing import Callable, List, Optional, Tuple, Union

from autostats.autostats import AutoStat
from sklearn.datasets import load_iris



class TestAutoTest:

    # we do remove NaN variables, duplicates, infinitive numbers
    def test_preprocessing(self, 
                           generate_df_NaN:Callable, 
                           generate_df_dupl:Callable, 
                           generate_df_inf:Callable,
                           generate_df_labels_test:Callable):
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

        with pytest.raises(
            AssertionError,
            match="The dataset is empty",
        ):
            NaN_all = stat.preprocessing(dataset = dataset_NaN_all, labels = "labels")

        # 3. no NaN, should do nothing
        dataset_no_NaN = generate_df_NaN(num_rows=100, num_cols=10, num_missing=0)
        df_preproc3 = stat.preprocessing(dataset = dataset_no_NaN,
                                        labels = "labels") 

        testing.assert_frame_equal(dataset_no_NaN, df_preproc3)
        assert dataset_no_NaN.isnull().sum().sum() == 0
        assert df_preproc3.isnull().sum().sum() == 0

        # 4. no duplicates, should do nothing
        dataset_no_dupl = generate_df_dupl(num_rows=100, num_cols=10, frac_dupl=0)
        df_preproc4 = stat.preprocessing(dataset = dataset_no_dupl,
                                        labels = "labels")
        
        testing.assert_frame_equal(dataset_no_dupl, df_preproc4)

        # 5. 10% of duplicates, should remove duplicates
        dataset_dupl = generate_df_dupl(num_rows=100, num_cols=10, frac_dupl=0.1)
        df_preproc5 = stat.preprocessing(dataset = dataset_dupl,
                                        labels = "labels")
        
        assert len(dataset_dupl) > len(df_preproc5)
        assert df_preproc5.shape == (100, 11)

        # 6. Exactly one duplicate
        dataset_dupl1 = generate_df_dupl(num_rows=100, num_cols=10, frac_dupl=0.01)
        df_preproc6 = stat.preprocessing(dataset = dataset_dupl1,
                                        labels = "labels")
        
        assert len(dataset_dupl1) == 101
        assert len(df_preproc6) == 100

        # 7. All rows have a duplicate 
        dataset_dupl_all = generate_df_dupl(num_rows=100, num_cols=10, frac_dupl=1)
        df_preproc7 = stat.preprocessing(dataset = dataset_dupl_all,
                                        labels = "labels")
        
        assert len(dataset_dupl_all) == 200
        assert len(df_preproc7) == 100

        # 8. check for infinitive numbers
        dataset_inf = generate_df_inf(num_rows=100, num_cols=10, num_inf=10)
        df_preproc8 = stat.preprocessing(dataset = dataset_inf,
                                        labels = "labels")
        
        assert len(dataset_inf) > len(df_preproc8)
        assert any(dataset_inf.isin([np.inf, -np.inf])) == True

        # TODO: there is some error here, debug
        # assert any(df_preproc8.isin([np.inf, -np.inf])) == False

        # # 9. Dataset with one row or with 1 column
        dataset_1row = generate_df_NaN(num_rows=1, num_cols=10, num_missing=0)

        with pytest.raises(
            AssertionError,
            match="The dataset has only one row",
        ):
            df_one_row = stat.preprocessing(dataset = dataset_1row, labels = "labels")
        
        # 10. dataset with one unique dependent variable
        dataset_1_label = generate_df_labels_test(num_rows=100, num_cols=10, labels=["label1"])
        with pytest.raises(
            AssertionError,
            match="Not enough samples in dataset, statistical tests are not appliable",
        ):
            df_one_label = stat.preprocessing(dataset = dataset_1_label, labels = "labels") 

        # 11. Dependent variable is not not categorical (integer or string)
        dataset_labels_test1 = generate_df_labels_test(num_rows=100, num_cols=10, labels=[1.1, 1.3, 110.142])
        with pytest.raises(
            AssertionError,
            match="Not categorical variables for groups: labels are neither strings nor integers",
        ):
            df_one_label = stat.preprocessing(dataset = dataset_labels_test1, labels = "labels") 


        dataset_labels_test2 = generate_df_labels_test(num_rows=100, num_cols=10, labels=[1, 1, 110.142, "STR"])
        with pytest.raises(
            AssertionError,
            match="Not categorical variables for groups: labels are neither strings nor integers",
        ):
            df_one_label = stat.preprocessing(dataset = dataset_labels_test2, labels = "labels") 

        # 11. Dependent variable is integer
        dataset_labels_test3 = generate_df_labels_test(num_rows=100, num_cols=10, labels=[0, 1, 2])
        df_label_int = stat.preprocessing(dataset = dataset_labels_test3, labels = "labels")

        testing.assert_frame_equal(dataset_labels_test3, df_label_int)

        # 12. Dependent variable is string
        dataset_labels_test4 = generate_df_labels_test(num_rows=100, num_cols=10, labels=["str1", "str2", "str3"])
        df_label_str = stat.preprocessing(dataset = dataset_labels_test4, labels = "labels")

        testing.assert_frame_equal(dataset_labels_test4, df_label_str)