# ##################################################
# # Rostyslav Makarenko 09/2023 ####################
# # all rights reserved ############################
# ##################################################


import os
import shutil
import numpy as np
import pandas as pd
from pandas import testing
import pytest

from typing import Callable, List, Optional, Tuple, Union

from autostats.autostats import GroupComparisonAnalyzer
from sklearn.datasets import load_iris

from scipy import stats

class TestGroupComparisonAnalyzer:

    def test_create_comb(self, generate_df_labels_test:Callable):
        # 1. common case: one pair of categories
        df = generate_df_labels_test(num_rows=100, num_cols=10, labels = ["Category 1", "Category 2"])
        stat = GroupComparisonAnalyzer(dataset=df, labels="labels")
        df_comb = stat.create_comb()
        assert len(df_comb) == 1
        assert len(df_comb[0]) == 2

        # extreme case: one category
        df2 = generate_df_labels_test(num_rows=100, num_cols=10, labels = ["Category 1"])

        with pytest.raises(
            AssertionError,
            match="no group combinations were created",
        ):
            df_comb = GroupComparisonAnalyzer(dataset=df2, labels="labels").create_comb()
        

    def test_normality_test(self, generate_df_norm_labels:Callable, generate_df_NaN:Callable):
        """
        Test with generated normal and non-normal distributed data"""

        # 1. Normal distribution < 50 observations for both categories
        df_norm1 = generate_df_norm_labels(n_obs_category1 = 40, n_obs_category2 = 30)
        stat = GroupComparisonAnalyzer(dataset=df_norm1, labels="labels")

        ##########################################################################################

        # we check if function chooses the right test for normality depending on the number of observations

        #  do we choose shapiro test for normality if number of observations < 50
        df_norm_tested1 = stat.perform_normality_test(df_norm1.loc[df_norm1["labels"] == "Category 1", "value"])     
        assert df_norm_tested1 == stats.shapiro(df_norm1.loc[df_norm1["labels"] == "Category 1", "value"])[1]

        #  do we choose ks test for normality if number of observations > 50
        df_norm1 = generate_df_norm_labels(n_obs_category1 = 60, n_obs_category2 = 30)
        df_norm_tested1 = stat.perform_normality_test(df_norm1.loc[df_norm1["labels"] == "Category 1", "value"])
        assert df_norm_tested1 == stats.kstest(df_norm1.loc[df_norm1["labels"] == "Category 1", "value"], "norm")[1]

        # do we choose shapiro test for normality if number of observations = 50
        df_norm1 = generate_df_norm_labels(n_obs_category1 = 50, n_obs_category2 = 30)
        df_norm_tested1 = stat.perform_normality_test(df_norm1.loc[df_norm1["labels"] == "Category 1", "value"])
        assert df_norm_tested1 == stats.shapiro(df_norm1.loc[df_norm1["labels"] == "Category 1", "value"])[1]

    def test_group_comparison_norm(self, generate_df_norm_labels:Callable):
        
        # 1 test two normal distributions
        df_norm = generate_df_norm_labels()

        stat = GroupComparisonAnalyzer(dataset=df_norm, labels="labels")

        results = []
        result = {
                    'Group 1': "Category 1",
                    'Group 2': "Category 2",
                    'Feature': "value",
                    'both are norm dist': 1.0
            }
        results.append(result)

        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot_table(values='both are norm dist', index=['Group 1', 'Group 2'], columns='Feature')

        pd.testing.assert_frame_equal(pivot_df, stat.group_comparison_norm())

         
         # check a case both a non-normal

         # check a case: one is normal and another is non-normal