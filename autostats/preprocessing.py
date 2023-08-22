##################################################
# Rostyslav Makarenko 09/2023 ####################
# all rights reserved ############################
##################################################

import pandas as pd
import numpy as np


# to prepare data for statistical analysis


class Preprocess:
    def __init__(
        self, 
        dataset: pd.DataFrame = None, 
        labels: str = None
    ) -> pd.DataFrame:
        self.dataset = dataset
        self.labels = labels


    def drop_non_numeric_columns(self, dataset:pd.DataFrame, labels:str) -> pd.DataFrame:
        """drop all non-numeric columns from the dataset for statistical analysis

        Args:
            dataset (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        columns_to_drop = []

        for column in dataset.columns:
            if column != labels:
                if not pd.api.types.is_numeric_dtype(dataset[column]):
                    columns_to_drop.append(column)

        dataset = dataset.drop(columns=columns_to_drop)
        return dataset

    def preprocessing(self, dataset:pd.DataFrame, labels:str) -> pd.DataFrame:
        """Cleaning the dataframe from missing values, duplicates, and infinite values,
        and dropping the labels column.

        Args:
            dataset (pd.DataFrame): _description_
            labels (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        
        # drop non-numeric columns
        dataset = self.drop_non_numeric_columns(dataset=dataset, labels=labels)

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
        # drop non-numeric columns
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
 
        return dataset