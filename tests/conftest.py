import pytest
import pandas as pd
import numpy as np
import random

from typing import Callable, List, Optional, Tuple, Union


@pytest.fixture
def generate_df_NaN() -> Callable[..., pd.DataFrame]:
    def _generate_df_NaN(num_rows:int, num_cols:int, num_missing:int) -> pd.DataFrame:
        """ creates a dataframe with random values for given number of rows and columns and including
        given number of missing values

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

@pytest.fixture
def generate_df_dupl() -> Callable[..., pd.DataFrame]:
    def _generate_df_dupl(num_rows:int, num_cols:int, frac_dupl:float) -> pd.DataFrame:
        """ creates a dataframe with random values for given number of rows and columns and 
        including ducplicates rows with given fraction

        Args:
            num_rows (int): _description_
            num_cols (int): _description_
            num_missing (int): _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = np.random.uniform(low=1, high=100, size=(num_rows, num_cols))
        df = pd.DataFrame(data)
        df["labels"] = random.choices(["label1", "label2"], k=len(df))

        duplicates = df.sample(frac=frac_dupl)

        df_dupl = pd.concat([df, duplicates])
        return df_dupl
    return _generate_df_dupl
    
@pytest.fixture
def generate_df_inf() -> Callable[..., pd.DataFrame]:
    def _generate_df_inf(num_rows:int, num_cols:int, num_inf:float) -> pd.DataFrame:
        """ creates a dataframe with random values for given number of rows and columns and including infinities values

        Args:
            num_rows (int): _description_
            num_cols (int): _description_
            num_missing (int): _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = np.random.uniform(low=1, high=100, size=(num_rows, num_cols))

        mask = np.random.choice([True, False], size=data.shape, p=[num_inf/(num_rows*num_cols), 1-num_inf/(num_rows*num_cols)])
        data[mask] = random.choices([np.inf, -np.inf])

        df_inf = pd.DataFrame(data)
        df_inf["labels"] = random.choices(["label1", "label2"], k=len(df_inf))


        return df_inf
    return _generate_df_inf


@pytest.fixture
def generate_df_labels_test() -> Callable[..., pd.DataFrame]:
    def _generate_df_labels_test(num_rows:int, num_cols:int, labels:list) -> pd.DataFrame:
        """generates dataframe with random values and given number of labels (to create groups)

        Args:
            num_rows (int): _description_
            num_cols (int): _description_
            num_missing (int): _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = np.random.uniform(low=1, high=100, size=(num_rows, num_cols))
        df_one_label = pd.DataFrame(data)
        df_one_label["labels"] = random.choices(labels, k=len(df_one_label))

        return df_one_label
    return _generate_df_labels_test



@pytest.fixture
def generate_df_norm_labels():
    def _generate_df_norm_labels(mean_category1:int=100,
                                 mean_category2:int=1000,
                                 scale_category1:int=10, 
                                 scale_category2:int=200,
                                 n_obs_category1:int=30, 
                                 n_obs_category2:int=50) -> pd.DataFrame:
        """_summary_

            Returns:
                _type_: _description_
            """

            # Define the number of observations for each category
        n_obs_category1 = n_obs_category1
        n_obs_category2 = n_obs_category2

            # Generate values from a normal distribution for category 1
        category1 = np.random.normal(loc=mean_category1, scale=scale_category1, size=n_obs_category1)

            # Generate values from a normal distribution for category 2
        category2 = np.random.normal(loc=mean_category2, scale=scale_category2, size=n_obs_category2)

            # Create a DataFrame with the generated values and categories
        df = pd.DataFrame({
            'value': np.concatenate([category1, category2]),
            'labels': ['Category 1'] * n_obs_category1 + ['Category 2'] * n_obs_category2
        })

        return df
    return _generate_df_norm_labels


# @pytest.fixture
# def generate_norm_res():
