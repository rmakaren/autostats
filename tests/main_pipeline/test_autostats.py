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

        # extreme case: no categories
        df = generate_df_labels_test(num_rows=100, num_cols=10, labels = [])
        stat = GroupComparisonAnalyzer(dataset=df, labels="labels")
        df_comb = stat.create_comb()
        assert len(df_comb) == 0
        

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
 