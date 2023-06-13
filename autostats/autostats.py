##################################################
# Rostyslav Makarenko 09/2023 ####################
# all rights reserved ############################
##################################################

import os
import copy
import pandas as pd
from scipy import stats
from typing import List, Dict, Union, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pingouin as pg
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols

import warnings
import time


warnings.filterwarnings("ignore")


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import seaborn as sns

from sklearn.datasets import load_iris


import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
import pingouin as pg

from typing import List, Dict, Callable, Tuple


class GroupComparisonAnalyzer:
    def __init__(self, dataset, labels):
        self.dataset:pd.DataFrame = dataset
        self.labels:str = labels
        self.combinations = self.create_comb()

    def create_comb(self) -> List[Tuple[str, str]]:
        """helper function to create all pairwise combinations of groups and features
        for group comparison

        Returns:
            List[tuple]: list of the tuples
        """
        groups = self.dataset[self.labels].unique()
        combinations = [(groups[i], groups[j]) for i in range(len(groups)) for j in range(i+1, len(groups))]
        return combinations

    def perform_normality_test(self, data:pd.Series) -> float:
        """Choose test for normality depending on the data size: either Shapiro-Wilk test or
        Kogmolorov-Smirnov test and perform that test

        Args:
            data (pd.DataFrame): pandas Series

        Returns:
            Callable: p-value as a results of the test for normality
        """
        if len(data) <= 50:
            return stats.shapiro(data)[1]
        else:
            return stats.kstest(data, cdf='norm')[1]

    def group_comparison_norm(self) -> pd.DataFrame:
        """performs test for normality on the given dataset with the corresponds to the groups of interest
        and checks if all groups for given feature pass the test for normality

        Returns:
            _type_: a pivot table, pandas dataframe to 
        """
        results = []

        for combination in self.combinations:
            group1, group2 = combination

            for feature in self.dataset.columns.drop(self.labels):
                group1_data = self.dataset[self.dataset[self.labels] == group1][feature]
                group2_data = self.dataset[self.dataset[self.labels] == group2][feature]

                p_value = any(pd.Series([self.perform_normality_test(group1_data),
                                         self.perform_normality_test(group2_data)]) < 0.05)

                result = {
                    'Group 1': group1,
                    'Group 2': group2,
                    'Feature': feature,
                    'p-value': p_value
                }
                results.append(result)

        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot_table(values='p-value', index=['Group 1', 'Group 2'], columns='Feature')
        print(pivot_df)
        return pivot_df

    def perform_variance_test(self, group1_data, group2_data, center:str) -> pd.DataFrame:
        """_summary_

        Args:
            group1_data (_type_): _description_
            group2_data (_type_): _description_
            center (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        return stats.levene(group1_data, group2_data, center=center)[1]

    def group_comparison_var(self, norm_test:pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            norm_test (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        results = []

        for combination in self.combinations:
            group1, group2 = combination

            for feature in self.dataset.columns.drop(self.labels):
                norm_value = norm_test.loc[combination][feature]
                center = "mean" if norm_value == 1 else "median"

                group1_data = self.dataset[self.dataset[self.labels] == group1][feature]
                group2_data = self.dataset[self.dataset[self.labels] == group2][feature]

                p_value = self.perform_variance_test(group1_data, group2_data, center)

                result = {
                    'Group 1': group1,
                    'Group 2': group2,
                    'Feature': feature,
                    'p-value': p_value
                }
                results.append(result)

        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot_table(values='p-value', index=['Group 1', 'Group 2'], columns='Feature')
        print(pivot_df)
        return pivot_df



    def choose_stat_test(self, norm_test: pd.DataFrame, var_test: pd.DataFrame, dependence="independent") -> pd.DataFrame:
        """_summary_

        Args:
            norm_test (pd.DataFrame): _description_
            var_test (pd.DataFrame): _description_
            dependence (str, optional): _description_. Defaults to "independent".

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """

        results = []
        
        for combination in self.combinations:
            group1, group2 = combination
            
            for feature in self.dataset.columns.drop(self.labels):
                norm_value = norm_test.loc[combination][feature]
                var_value = var_test.loc[combination][feature]

                group1_data = self.dataset[self.dataset[self.labels] == group1][feature]
                group2_data = self.dataset[self.dataset[self.labels] == group2][feature]
                print(group1_data, group2_data)
                
                if norm_value == 1 and var_value < 0.05 and dependence == "independent":
                    stat_test = stats.f_oneway(group1_data, group2_data)[1]
                elif norm_value == 1 and var_value < 0.05 and dependence == "paired":
                    stat_test = sm.stats.anova.AnovaRM(self.dataset, feature, subject=self.labels, within=[group1, group2]).fit()
                elif norm_value == 1 and var_value >= 0.05 and dependence == "independent":
                    stat_test = pg.welch_anova(self.dataset, dv=feature, between=self.labels).loc[0, 'p-unc']
                elif norm_value == 1 and var_value >= 0.05 and dependence == "paired":
                    stat_test = sm.stats.anova_lm(sm.stats.OLS(self.dataset[group1][feature], sm.tools.add_constant(self.dataset[group2][feature]))).F[0]
                elif norm_value == 0 and var_value < 0.05 and dependence == "independent":
                    stat_test = stats.kruskal(group1_data, group2_data)[1]
                elif norm_value == 0 and var_value < 0.05 and dependence == "paired":
                    stat_test =  stats.friedmanchisquare(group1_data, group2_data)[1]
                elif norm_value == 0 and var_value >= 0.05 and dependence == "independent":
                    stat_test = stats.median_test(group1_data, group2_data)[1]
                elif norm_value == 0 and var_value >= 0.05 and dependence == "paired":
                    stat_test = stats.wilcoxon(group1_data, group2_data)[1]
                else:
                    raise ValueError("something went wrong")
                
                result = {
                    'Group 1': group1,
                    'Group 2': group2,
                    'Feature': feature,
                    'stat-test': stat_test
                }
                results.append(result)

        results_df = pd.DataFrame(results)
        print(results_df)
        pivot_df = results_df.pivot_table(values='stat-test', index=['Group 1', 'Group 2'], columns='Feature')
        print(pivot_df)
        return pivot_df
    
    def run_group_comparison(self, dependence="independent") -> pd.DataFrame:
        """_summary_

        Args:
            norm_test (pd.DataFrame): _description_
            var_test (pd.DataFrame): _description_
            dependence (str, optional): _description_. Defaults to "independent".

        Returns:
            pd.DataFrame: _description_
        """
        norm_test_result = self.group_comparison_norm()
        var_test_result = self.group_comparison_var(norm_test=norm_test_result)
        res = self.choose_stat_test(norm_test=norm_test_result, var_test=var_test_result, dependence=dependence)
        return res

if __name__ == "__main__":
    df_iris = sns.load_dataset("iris")
    analyzer = GroupComparisonAnalyzer(dataset=df_iris, labels="species")
    norm_test_result = analyzer.group_comparison_norm()
    var_test_result = analyzer.group_comparison_var(norm_test=norm_test_result)
    group_test = analyzer.choose_stat_test(norm_test=norm_test_result, var_test=var_test_result)
    