import os
import numpy as np
import pandas as pd
import pytest


from typing import Callable, List, Optional, Tuple, Union

os.chdir("/media/rostyslav/Toshiba/Projects/autostats/02-scripts/autostats")

from autostats.autostats import AutoStat
from sklearn.datasets import load_iris

import conftest
# 
# housing = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/housing.csv")

# iris = load_iris()
# #iris.keys()

# df_iris = pd.DataFrame(
#     data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
# )
# df_iris["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
# df_iris = df_iris.drop("target", axis=1)


class TestAutoTest:
    def test_autostat(self, missing_df:Callable):
        dataset = missing_df(num_rows=100, num_cols=10, num_missing=0.1)
        stat = AutoStat()
        df_preproc = stat.preprocessing(dataset=dataset)
        pass
        # assert df_preproc.isnull().sum().sum() == 0

