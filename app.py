import  os
import pandas as pd
import numpy as np

from autostats.preprocessing import Preprocess
from autostats.autostats import GroupComparisonAnalyzer

dataset = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/heart.csv", sep=",")

preprocess = Preprocess(dataset=dataset, labels="output")

dataset_prep = preprocess.preprocessing(dataset=dataset, labels="output")
dataset_prep2 = dataset_prep.drop("caa", axis=1)
stats = GroupComparisonAnalyzer(dataset=dataset_prep2, labels="output", output_path="/media/rostyslav/Toshiba/Projects/autostats/02-output/")
stats.run_group_comparison()

#%%

import  os
import pandas as pd
import numpy as np

from autostats.preprocessing import Preprocess
from autostats.autostats import GroupComparisonAnalyzer

dataset = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/AB_NYC_2019_for_visualisation.csv", sep=",")

preprocess = Preprocess(dataset=dataset, labels="neighbourhood_group")

dataset_prep = preprocess.preprocessing(dataset=dataset, labels="neighbourhood_group")
# dataset_prep2 = dataset_prep.drop("caa", axis=1)
stats = GroupComparisonAnalyzer(dataset=dataset_prep, labels="neighbourhood_group", output_path="/media/rostyslav/Toshiba/Projects/autostats/02-output/")
stats.run_group_comparison()
