import os
import pandas as pd
from scipy import stats
from typing import List, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        self, dataset: pd.DataFrame = None, labels: str = None, output_dir: str = None
    ) -> None:

        self.dataset = dataset
        self.labels = labels
        self.output_dir = output_dir

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

        elif len(dataset[labels].unique()) == 2 and len(dataset) >= 5:
            print("comparing two groups")
            return "comparing two groups"
        elif len(dataset[labels].unique()) > 2 and len(dataset) >= 5:
            print("comparing multiple groups")
            return "comparing multiple groups"

    # def f_test(self, dataset, labels) -> float:
    #     """an F-Test in Python
    #     from https://www.geeksforgeeks.org/how-to-perform-an-f-test-in-python/

    #     Args:
    #         group1 (_type_): _description_
    #         group2 (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """

    #     x = np.array(dataset[labels] == dataset[labels].unique()[0])
    #     y = np.array(dataset[labels] == dataset[labels].unique()[1])
    #     f = np.var(dataset[labels] == dataset[labels].unique()[0],
    #                 ddof=1)/np.var(dataset[labels] == dataset[labels].unique()[1], ddof=1)
    #     nun = x.size-1
    #     dun = y.size-1
    #     p_value = 1-stats.f.cdf(f, nun, dun)
    #     return f, p_value

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

        for observation in unique_observations:
            ix_a = dataset[labels] == observation
            for x in columns:
                norm_test[observation + "_shapiro_" + x] = stats.shapiro(
                    dataset[x][ix_a]
                )[0]
                norm_test[observation + "_kolmogorov_" + x] = stats.kstest(
                    dataset[x][ix_a], stats.norm.cdf
                )[0]

        return pd.DataFrame.from_dict(norm_test, orient="index")

    def variance_bartlett(self, dataset: pd.DataFrame, labels: str) -> pd.DataFrame:
        """_summary_

        Args:
            dataset (pd.DataFrame): _description_
            labels (str): _description_
            STAT_TYPE (str): _description_

        Returns:
            Dict: _description_
        """
        all_variance = {}
        item_list = list(dataset[labels].unique())
        for column in dataset.columns:
            if column != labels:
                variances_test = stats.bartlett(
                    dataset[dataset[labels] == item_list[0]][column],
                    dataset[dataset[labels] == item_list[1]][column],
                )
                all_variance[column] = variances_test[0]
        return pd.DataFrame.from_dict(all_variance, orient="index")

    def variance_levene(self, dataset: pd.DataFrame, labels: str) -> Dict:
        """_summary_

        Args:
            dataset (pd.DataFrame): _description_
            labels (str): _description_

        Returns:
            Dict: _description_
        """

        all_variance = {}
        item_list = list(dataset[labels].unique())
        for column in dataset.columns:
            if column != labels:
                variances_test = stats.levene(
                    dataset[dataset[labels] == item_list[0]][column],
                    dataset[dataset[labels] == item_list[1]][column],
                )
                all_variance[column] = variances_test[0]
        return pd.DataFrame.from_dict(all_variance, orient="index")

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
        normality = self.normality_test(dataset, labels)
        is_normal_distribution = pd.Series(normality[0] > 0.05).all()

        if STAT_TYPE == "comparing two groups" and is_normal_distribution:
            variance = self.variance_bartlett(dataset, labels)
        elif STAT_TYPE == "comparing two groups" and not is_normal_distribution:
            variance = self.variance_levene(dataset, labels)

        item_list = list(dataset[labels].unique())
        for column in dataset.columns:
            if column not in labels:
                if is_normal_distribution and STAT_TYPE == "comparing two groups":
                    describe_stats.loc[column, "pvalue"] = round(
                        stats.ttest_ind(
                            dataset[dataset[labels] == item_list[0]][column],
                            dataset[dataset[labels] == item_list[1]][column],
                            equal_var=int(variance.loc[column]) > 0.05,
                        )[1],
                        3,
                    )
                    describe_stats.loc[
                        (column, "mean"), "test_type"
                    ] = "t-test independent"
                if not is_normal_distribution and STAT_TYPE == "comparing two groups":
                    describe_stats.loc[column, "pvalue"] = round(
                        stats.mannwhitneyu(
                            dataset[dataset[labels] == item_list[0]][column],
                            dataset[dataset[labels] == item_list[1]][column],
                        )[1],
                        3,
                    )
                    describe_stats.loc[
                        (column, "mean"), "test_type"
                    ] = "Mann-Whitney U test"
                if not is_normal_distribution and STAT_TYPE == "comparing two groups":
                    describe_stats.loc[column, "pvalue"] = round(
                        stats.mannwhitneyu(
                            dataset[dataset[labels] == item_list[0]][column],
                            dataset[dataset[labels] == item_list[1]][column],
                        )[1],
                        3,
                    )
                    describe_stats.loc[
                        (column, "mean"), "test_type"
                    ] = "Mann-Whitney U test"
                if STAT_TYPE == "comparing multiple groups" and is_normal_distribution:
                    describe_stats.loc[(column, "mean"), "pvalue"] = stats.f_oneway(
                        *(
                            dataset.loc[dataset[labels] == group, column]
                            for group in dataset[labels].unique()
                        )
                    )[1]
                    describe_stats.loc[(column, "mean"), "test_type"] = "one-way Anova"
                if (
                    STAT_TYPE == "comparing multiple groups"
                    and not is_normal_distribution
                ):
                    describe_stats.loc[(column, "mean"), "pvalue"] = stats.kruskal(
                        *(
                            dataset.loc[dataset[labels] == group, column]
                            for group in dataset[labels].unique()
                        )
                    )[1]
                    describe_stats.loc[
                        (column, "mean"), "test_type"
                    ] = "Kruskal-Wallis H-test"

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
