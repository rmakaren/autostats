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

dataset = pd.read_csv("/media/rostyslav/Toshiba/Projects/genomic_vision/QC/2nd_analysis/data_for_analysis_prerop.csv", sep=",")

preprocess = Preprocess(dataset=dataset, labels="Statut_lamelle")

# "Advancing", "ADV_Max-Min", "Receding", "REC_Max-Min", 
# "DSMulti_level", "BC_level", "H_level_", "TS_DNA_fix_level", "Score_QC",
# "Density", "Density_perc", "Waviness_level", "RCD_level",

dataset = dataset[["CA", "Statut_lamelle"]]

dataset_prep = preprocess.preprocessing(dataset=dataset, labels="Statut_lamelle")
# dataset_prep2 = dataset_prep.drop("caa", axis=1)
stats = GroupComparisonAnalyzer(dataset=dataset_prep, labels="Statut_lamelle", output_path="/media/rostyslav/Toshiba/Projects/genomic_vision/QC/2nd_analysis/")
stats.run_group_comparison()

# %%
