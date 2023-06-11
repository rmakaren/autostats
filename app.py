import  os
import pandas as pd
import numpy as np

from autostats.preprocessing import Preprocess
from autostats.autostats import GroupComparisonAnalyzer



dataset = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/Iris.csv", sep=",")

preprocess = Preprocess(dataset=dataset, labels="Species")
stats = GroupComparisonAnalyzer(dataset=dataset, labels="Species")

dataset = preprocess.preprocessing(dataset=dataset, labels="Species")
print(dataset)
res = stats.run_group_comparison()
print(res)