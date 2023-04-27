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
        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.drop_duplicates()
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.dropna()
        # dataset = dataset.drop(columns=[labels])
        dataset = dataset.reset_index(drop=True)

        # check if the dataset is empty
        if dataset.empty:
            print("The dataset is empty")
            return None
        
        # check if the dataset has only one column
        elif len(dataset.columns) == 1:
            print("The dataset has only one column")
            return None

        # check if the dataset has only one row
        elif len(dataset) == 1:
            print("The dataset has only one row")
            return None
        
        # check if the dataset has only one unique value
        elif len(dataset[labels].unique()) == 1:
            print("The dataset has only one unique value")
            return None
        
        # check if labels are either strings or integers
        elif not all(isinstance(x, (str, int)) for x in dataset[labels]):
            print("Not categorical variables for groups: labels are neither strings nor integers")
            return None

        return dataset

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


    def normality_test(self, dataset, labels, output_dir) -> pd.DataFrame:

        def normality_tests(dataset):
            results = {}
            for col in dataset.drop(labels, axis = 1).columns:
                if len(dataset)<= 50:
                    print(f"using shapiro test for {col}")
                    _, shapiro_pval = stats.shapiro(dataset[col])
                    results[col] = {"shapiro": shapiro_pval}
                elif len(dataset)>50:
                    print(f"using ks test for {col}")
                    _, ks_pval = stats.kstest(dataset[col], "norm")
                    results[col] = {"ks": ks_pval}
            return pd.DataFrame(results)

        norm_res = dataset.groupby([labels]).apply(normality_tests)

        # visualise results of normality tests with qqplots
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

    def variance_test(self, dataset:pd.DataFrame, labels, norm_res:pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            dataset (pd.DataFrame): _description_
            labels (_type_): _description_
            normality (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        
        results = {}
        for col in dataset.drop(labels, axis = 1).columns:
            if pd.Series(normality[0] > 0.05).all():
                if len(dataset) <= 50:
                    print(f"using levene's test for {col}")
                    _, levene_pval = stats.levene(dataset[col])
                    results[col] = {"levene": levene_pval}
                elif len(dataset) > 50:
                    print(f"using bartlett's test for {col}")
                    _, bartlett_pval = stats.bartlett(dataset[col])
                    results[col] = {"barlett": bartlett_pval}
                elif not pd.Series(normality[0] < 0.05).any():
                    print("Brown-Forsythe test for variance homogeneity")


            for col in dataset.drop(labels, axis = 1).columns:
                if len(dataset)<= 50:
                    print(f"using shapiro test for {col}")
                    _, shapiro_pval = stats.shapiro(dataset[col])
                    results[col] = {"shapiro": shapiro_pval}
                elif len(dataset)>50:
                    print(f"using ks test for {col}")
                    _, ks_pval = stats.kstest(dataset[col], "norm")
                    results[col] = {"ks": ks_pval}
            return pd.DataFrame(results)


    def define_stat_test(self, 
                         normality, 
                         variance, 
                         dependence:str = "independent", 
                         p_value: float = 0.05) -> Callable:
        """_summary_

        Args:
            normality (_type_): _description_
            variance (_type_): _description_
            dependence (str, optional): _description_. Defaults to "independent".
            p_value (float, optional): _description_. Defaults to 0.05.

        Returns:
            Callable: _description_
        """


        if pd.Series(normality[0] > p_value).all():
            if pd.Series(variance.iloc[1] > p_value).all():
                if dependence == "independent":
                    stat_test = stats.f_oneway
                elif dependence == "paired":
                    stat_test = AnovaRM 
            elif pd.Series(variance.iloc[1] < p_value).any():
                if dependence == "independent":
                    stat_test = pg.welch_anova
                elif dependence == "paired":
                    stat_test = sm.stats.anova_lm
        elif pd.Series(normality[0] < p_value).any():
            if pd.Series(variance.iloc[1] > p_value).all():
                if dependence == "independent":
                    stat_test = stats.kruskal
                elif dependence == "paired":
                    stat_test = stats.friedmanchisquare
            elif pd.Series(variance.iloc[1] < p_value).any():
                if dependence == "independent":
                    stat_test = stats.median_test
                elif dependence == "paired":
                    stat_test = stats.wilcoxon
        print("stat test", stat_test)
        return stat_test

    def make_stat_report(self, dataset:pd.DataFrame, labels:str, stat_test:Callable, output_dir:str) -> None:
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
            if stat_test != pg.welch_anova:
                describe_stats.loc[(column, "mean"), "pvalue"] = stat_test(
                    *(
                        dataset.loc[dataset[labels] == group, column]
                        for group in dataset[labels].unique()
                    )
                        )[1]
            elif stat_test == pg.welch_anova:
                describe_stats.loc[(column, "mean"), "pvalue"] = stat_test(
                    dataset, dv=column, between=labels
                )["p-unc"][0]

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

        STAT_TYPE = self.define_analysis_type(dataset, labels)
        if STAT_TYPE == None:
            return ("not enough samples, statistical tests are not applible")
        
        dataset = self.preprocessing(dataset, labels)

        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)

        normality = self.normality_test(dataset, labels, output_dir)
        variance = self.variance_test(dataset, labels, normality)
        # print(variance)
        s_test = self.define_stat_test(normality, variance, dependence = "independent")
        self.make_stat_report(dataset, labels, s_test, output_dir)
        print("--- %s seconds ---" % (time.time() - start_time))
        return normality



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