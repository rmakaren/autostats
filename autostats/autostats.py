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


class AutoStat:
    def __init__(
        self, 
        dataset: pd.DataFrame = None, 
        labels: str = None, 
        output_dir: str = None, 
        dependence:str = "independent"
    ) -> None:
        """_summary_

        Args:
            dataset (pd.DataFrame, optional): _description_. Defaults to None.
            labels (str, optional): _description_. Defaults to None.
            output_dir (str, optional): _description_. Defaults to None.
            dependence (str, optional): _description_. Defaults to "independent".
        """

        self.dataset = dataset
        self.labels = labels
        self.output_dir = output_dir
        self.dependence = dependence

    def preprocessing(self, dataset:pd.DataFrame, labels:str) -> pd.DataFrame:
        """Cleaning the dataframe from missing values, duplicates, and infinite values,
        and dropping the labels column.

        Args:
            dataset (pd.DataFrame): _description_
            labels (str): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # drop missing values, duplicates, and infinite values
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.drop_duplicates()
        # dataset = dataset.drop(columns=[labels])
        dataset = dataset.reset_index(drop=True)

        # drop missing values, duplicates, and infinite values, time-series
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.drop_duplicates()
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.select_dtypes(exclude=['datetime64'])
        dataset = dataset.reset_index(drop=True)

        # we remove the columns that have less than 5 unique values except the labels column
        dataset = dataset.loc[:, (dataset.nunique() >= 5) | (dataset.columns == labels)]

        # if after cleaning dataset is no more, return None
        assert type(dataset) != None, "The dataset is None, check the dataset"

        # check if the dataset is empty
        assert dataset.empty != True, "The dataset is empty"
        
        # check if the dataset has only one column
        assert len(dataset.columns) != 1, "The dataset has only one column"

        # check if the dataset has only one row
        assert len(dataset.index) != 1, "The dataset has only one row"
        
        # check if the dataset has enough data for analysis one unique value
        assert len(dataset[labels].unique()) != 1, "Not enough samples in dataset, statistical tests are not appliable"
        assert len(dataset) >= 5, "Not enough samples in dataset, statistical tests are not appliable"
        
        # check if labels are either strings or integers
        assert all(isinstance(x, (str, int)) for x in dataset[labels]), "Not categorical variables for groups: labels are neither strings nor integers"

        print("comparing two or more groups")
        return dataset


    def normality_test(self, dataset:pd.DataFrame, labels:str, output_dir:str) -> pd.DataFrame:
        """ Here we check the data for normality by QQplot visualisation and statistical tests
        per each feature we compare in the group. the output results for normality test are saved
        in .csv file 

        Args:
            dataset (_type_): _description_
            labels (_type_): _description_
            output_dir (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """

        def normality_tests(dataset:pd.DataFrame, labels=labels) -> pd.DataFrame:
            results = {}
            for col in dataset.columns.drop(labels):
                if len(dataset)<= 50:
                    norm_test = stats.shapiro
                    # print(f"using shapiro test for {col}")
                    _, shapiro_pval = norm_test(dataset[col])
                    results[col] = {"shapiro": shapiro_pval}
                elif len(dataset)>50:
                    norm_test = stats.kstest
                    # print(f"using ks test for {col}")
                    _, ks_pval = norm_test(dataset[col], "norm")
                    results[col] = {"ks": ks_pval}
            return pd.DataFrame(results)

        norm_res = dataset.groupby([labels]).apply(normality_tests)

        # visualise results of normality tests with QQplots
        norm_dir = os.path.join(output_dir, "normality_test")
        os.makedirs(norm_dir, exist_ok=True)

        unique_observations = dataset[labels].unique()
        columns = dataset.columns.tolist()
        columns.remove(labels)
        # calculate test statistics
        for observation in unique_observations:
            ix_a = dataset[labels] == observation
            for x in columns:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                res = stats.probplot(dataset[x][ix_a], dist ='norm', plot=ax)
                ax.set_title(f"qqplot_norm_{x}_{observation}")
                plt.savefig(os.path.join(norm_dir, f"qqplot_norm_{x}_{observation}.png"))
                plt.close()

        # save the results of normality test
        norm_res.to_csv(os.path.join(norm_dir, "norm_test.csv"))
        return norm_res


    def variance_test(self, dataset:pd.DataFrame, labels:str, norm_res:pd.DataFrame, output_dir:str) -> pd.DataFrame:
        """We perform test for equality of variances between groups 
        (both on normally and non-normally distributed data) 
        to choose an appropriate test for statistical analysis for group comparison. 
        1) There are less than 50 datapoints and data is normally distributed
            -> Levene's test for equality of variances
        2) There are more than 50 datapoints and data is normally distributed
            -> Bartlett's test test for equality of variances
        3) Data is not normally distributed
            -> "Brown-Forsythe test for variance homogeneity


        Args:
            dataset (pd.DataFrame): _description_
            labels (_type_): _description_
            norm_res (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        
        var_res = {}
        for column in dataset.columns.drop(labels):
            if all(norm_res[column] > 0.05):
                stat_test = stats.levene
                center = "mean"
            elif any(norm_res[column] < 0.05):
                stat_test = stats.levene
                center = "median"
            var_res[column] = round(stat_test(*(dataset.loc[dataset[labels] == group, column] 
                        for group in dataset[labels].unique()), center=center)[1], 7)
        
        var_res = pd.DataFrame.from_dict(var_res, orient='index', columns=["variance_test"]).T
        var_res.to_csv(path_or_buf=os.path.join(output_dir, "var_test.csv"))
        return var_res

    def choose_group_test(self, dataset:pd.DataFrame, labels:str, norm_res:pd.DataFrame, var_res:pd.DataFrame) -> Callable:
        """Choose a group test on the results of normality and variance tests

        Args:
            dataset (pd.DataFrame): _description_
            labels (str): _description_
            norm_res (pd.DataFrame): _description_
            var_res (pd.DataFrame): _description_
            output_dir (str): _description_
            dependence (str): _description_

        Returns:
            function: test function to perform pairwise comparison of groups
        """

        for column in dataset.columns.drop(labels):
            if all(norm_res[column] > 0.05) & (var_res.loc['variance_test'][column] > 0.05):
                print(column, "normal distribution, equal variance")
                if self.dependence == "independent":
                    group_test = stats.f_oneway
                elif self.dependence == "dependent":
                    group_test = AnovaRM
            elif all(norm_res[column] > 0.05) & (var_res.loc['variance_test'][column] < 0.05):
                print(column, "normal distribution, unequal variance")
                if self.dependence == "independent":
                    group_test = pg.welch_anova
                elif self.dependence == "dependent":
                    group_test = sm.stats.anova_lm
            elif any(norm_res[column] < 0.05) & (var_res.loc['variance_test'][column] > 0.05):
                print(column, "not normal distribution, equal variance")
                if self.dependence == "independent":
                    group_test = stats.kruskal
                if self.dependence == "dependent":
                    group_test = stats.friedmanchisquare
            elif any(norm_res[column] < 0.05) & (var_res.loc['variance_test'][column] < 0.05):
                print(column, "not normal distribution, unequal variance")
                if self.dependence == "independent":
                    group_test = stats.median_test
                elif self.dependence == "dependent":
                    group_test = stats.wilcoxon
            else:
                print(column, "something is wrong")
        return group_test


    def make_stat_report(self, group_test:Callable, dataset:pd.DataFrame, labels:str, output_dir:str) -> None:
        """_summary_

        Args:
            dataset (pd.DataFrame): _description_
            labels (_type_): _description_
            stat_test (Callable): _description_
            output_dir (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """

        describe_stats = round(
            pd.DataFrame(dataset.groupby([labels]).describe().transpose()), 3
        )
        for column in dataset.columns.drop(labels):
            if group_test in [stats.f_oneway, 
                              stats.kruskal, 
                              stats.friedmanchisquare, 
                              stats.median_test, 
                              stats.wilcoxon,
                              AnovaRM]:
                describe_stats.loc[(column, "mean"), "pvalue"] = group_test(
                    *(
                    dataset.loc[dataset[labels] == group, column]
                    for group in dataset[labels].unique()
                    )
                    )[1]
            elif group_test in [pg.welch_anova]:
                describe_stats.loc[(column, "mean"), "pvalue"] = group_test(
                    dataset, dv=column, between=labels, detailed=True
                ).loc[labels, "p-unc"]
            elif group_test in [sm.stats.anova_lm]:
                describe_stats.loc[(column, "mean"), "pvalue"] = sm.stats.anova_lm(
                    dataset, dv=column, between=labels
                    )["PR(>F)"][0]
            else:
                print(column, "something is wrong")
            sns.violinplot(data=dataset, x=labels, y=column)
            sns.violinplot(data=dataset, x=labels, y=column).set(
                    title=f'p-value = {describe_stats.loc[(column, "mean"), "pvalue"]}'
                )
            plt.savefig(os.path.join(output_dir, f"compplot_{column}.png"))
            plt.clf()

        describe_stats.to_csv(os.path.join(output_dir, f"comp_stats.csv"))


    def auto_stat_test(self, dataset:pd.DataFrame, labels:str, output_dir:str) -> None:
        """_summary_

        Args:
            dataset (pd.DataFrame): a dataset you would like to use for analysis
            labels (str): your categories you would like to use for stat analysis, name of the column

        Returns:
            pd.DataFrame: _description_
        """

        
        start_time = time.time()
        
        dataset = self.preprocessing(dataset, labels)

        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)

        norm_res = self.normality_test(dataset, labels, output_dir)
        var_res = self.variance_test(dataset, labels, norm_res, output_dir)
        group_test = self.choose_group_test(dataset, labels, norm_res, var_res)
        self.make_stat_report(group_test=group_test, dataset=dataset, labels = "species", output_dir=output_dir)
        print("--- %s seconds ---" % (time.time() - start_time))
        return norm_res



if __name__ == "main":

    print("starting")
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
    print("done")