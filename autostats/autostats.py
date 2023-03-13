import os
import pandas as pd
from scipy import stats
from typing import List, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pingouin as pg
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols

import warnings

warnings.filterwarnings("ignore")


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import seaborn as sns

from sklearn.datasets import load_iris


class AutoStat:
    def __init__(
        self, dataset: pd.DataFrame = None, 
        labels: str = None, 
        output_dir: str = None, 
        dependence:str = "independent"
    ) -> None:

        self.dataset = dataset
        self.labels = labels
        self.output_dir = output_dir
        self.dependence = dependence

    def define_analysis_type(self, dataset, labels) -> Union[str, None]:
        """Setting up statistical tests to check

        Args:
            dataset (_type_): _description_
            labels (_type_): _description_

        Returns:
            Union[str, None]: _description_
        """
        if len(dataset[labels].unique()) == 1 or len(dataset) < 5:
            print("not enough samples, statistical tests are not applible")
            return None

        elif len(dataset[labels].unique()) >= 2 and len(dataset) >= 5:
            print("comparing two or more groups")
            return "comparing two or more groups"

    def normality_test(self, dataset: pd.DataFrame, labels: str) -> pd.DataFrame:
        """_summary_

        Args:
            dataset (pd.DataFrame): _description_
            labels (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        norm_test = {}
        unique_observations = dataset[labels].unique()
        columns = dataset.columns.tolist()
        columns.remove(labels)

        # Define functions to calculate test statistics
        shapiro = lambda x: stats.shapiro(x)[0]
        kolmogorov = lambda x: stats.kstest(x, stats.norm.cdf)[0]

        # Apply functions to calculate test statistics for each column and observation
        for observation in unique_observations:
            ix_a = dataset[labels] == observation
            subset = dataset.loc[ix_a, columns]
            norm_test.update(subset.apply(lambda x: {
                observation + "_shapiro_" + x.name: shapiro(x),
                observation + "_kolmogorov_" + x.name: kolmogorov(x)
            }).to_dict())

        return pd.DataFrame.from_dict(norm_test, orient="index")

    def variance_test(self, dataset:pd.DataFrame, labels, normality) -> pd.DataFrame:
        """_summary_

        Args:
            dataset (pd.DataFrame): _description_
            labels (_type_): _description_
            normality (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """

        all_variance = {}
        item_list = list(dataset[labels].unique())
        if pd.Series(normality[0] > 0.05).all():
            variance_func = stats.bartlett
        else:
            variance_func = stats.levene

        for column in dataset.columns:
            if column != labels:
                variances_test = variance_func(
                    dataset[dataset[labels] == item_list[0]][column],
                    dataset[dataset[labels] == item_list[1]][column],
                )
                all_variance[column] = variances_test[0]
        return pd.DataFrame.from_dict(all_variance, orient="index")


    def define_stat_test(self, normality, variance, dependence:str = "independent", p_value: float = 0.05) -> function:
        """_summary_"""


        if pd.Series(normality[0] > p_value).all():
            if pd.Series(variance[0] > p_value).all():
                if dependence == "independent":
                    stat_test = stats.f_oneway
                elif dependence == "paired":
                    stat_test = AnovaRM 
            elif pd.Series(variance[0] < p_value).all():
                if dependence == "independent":
                    stat_test = pg.welch_anova
                elif dependence == "paired":
                    stat_test = sm.stats.anova_lm
        elif pd.Series(normality[0] < p_value).all():
            if pd.Series(variance[0] > p_value).all():
                if dependence == "independent":
                    stat_test = stats.kruskal
                elif dependence == "paired":
                    stat_test = stats.friedmanchisquare
            elif pd.Series(variance[0] < p_value).all():
                if dependence == "independent":
                    stat_test = stats.median_test
                elif dependence == "paired":
                    stat_test = stats.wilcoxon
        return stat_test

    def auto_stat_test(self, dataset, labels, output_dir) -> pd.DataFrame:
        """_summary_

        Args:
            dataset (pd.DataFrame): a dataset you would like to use for analysis
            labels (str): your categories you would like to use for stat analysis, name of the column

        Returns:
            pd.DataFrame: _description_
        """

        describe_stats = round(
            pd.DataFrame(dataset.groupby([labels]).describe().transpose()), 3
        )

        STAT_TYPE = self.define_analysis_type(dataset, labels)
        if STAT_TYPE == None:
            return ("not enough samples, statistical tests are not applible")
        normality = self.normality_test(dataset, labels)
        variance = self.variance_test(dataset, labels, normality)
        stat_test = self.define_stat_test(normality, variance, dependence = "independent")

        for column in dataset.columns:
            describe_stats.loc[(column, "mean"), "pvalue"] = stat_test(
                *(
                    dataset.loc[dataset[labels] == group, column]
                    for group in dataset[labels].unique()
                )
                    )[1]

        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        for column in dataset.columns:
            if column not in labels:
                sns.violinplot(data=dataset, x=labels, y=column)
                sns.violinplot(data=dataset, x=labels, y=column).set(
                    title=f'p-value = {describe_stats.loc[(column, "mean"), "pvalue"]}'
                )
                plt.savefig(os.path.join(output_dir, f"comp_stat_{column}.png"))
                plt.clf()

        describe_stats.to_csv(os.path.join(output_dir, f"comp_stat.csv"))
        return describe_stats, normality


############################################""#
# EVERYTHING BELOW HERE IS TO TEST THE CODE #
#############################################

if __name__ == "main":

    iris = load_iris()
    df_iris = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    df_iris["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    # we start from simpler: assume there are only two species and only two valuables to compare, e.g. 2x2 matrix
    df_iris = df_iris[df_iris["species"] != "setosa"]
    df_iris = df_iris[["sepal length (cm)", "sepal width (cm)", "species"]]

    stat_test = AutoStat()
    stat_test.auto_stat_test(dataset=df_iris, labels="species")
