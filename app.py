# %%

# import  os
# import pandas as pd
# import numpy as np

# from autostats.preprocessing import Preprocess
# from autostats.autostats import GroupComparisonAnalyzer


# # iris dataset
# dataset = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/Iris.csv", sep=",")
# dataset = dataset.drop("Id", axis=1)
# dataset = dataset[dataset["Species"] != "Iris-setosa"]


# preprocess = Preprocess(dataset=dataset, labels="Species")
# stats = GroupComparisonAnalyzer(dataset=dataset, labels="Species")

# dataset = preprocess.preprocessing(dataset=dataset, labels="Species")
# res = stats.run_group_comparison()

# %%
# heart disease dataset


# THIS IS FAILED EXAMPLE

import  os
import pandas as pd
import numpy as np


from autostats.preprocessing import Preprocess
from autostats.autostats import GroupComparisonAnalyzer


dataset = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/heart.csv", sep=",")
# dataset = pd.read_csv("/home/rostyslav/Projects/STAT_project/01-data/heart.csv", sep=",")



preprocess = Preprocess(dataset=dataset, labels="output")


dataset_prep = preprocess.preprocessing(dataset=dataset, labels="output")
dataset_prep2 = dataset_prep.drop("caa", axis=1)
# dataset_prep2.remove_unused_categories()
stats = GroupComparisonAnalyzer(dataset=dataset_prep2, labels="output", output_path="/media/rostyslav/Toshiba/Projects/autostats/02-output/")
res = stats.run_group_comparison(adjustment="bonferroni")

#%%

# from statsmodels.sandbox.stats.multicomp import multipletests

# res.values.flatten()

# # p_adjusted = list(multipletests(res.values.flatten(), method='bonferroni')[1])

# [ '%.5f' % elem for elem in list(multipletests(res.values.flatten(), method='bonferroni')[1])]

# [ '%.5f' % elem for elem in list(multipletests(res.values.flatten(), method='bonferroni')[1])]


# %%

# wine dataset

# import  os
# import pandas as pd
# import numpy as np


# from autostats.preprocessing import Preprocess
# from autostats.autostats import GroupComparisonAnalyzer

# df1 = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/winequality-red.csv", sep=";")
# df1["type"] = "red"
# df2 = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/winequality-white.csv", sep=";")
# df2["type"] = "white"

# df = pd.concat([df1, df2], axis=0)

# preprocess = Preprocess(dataset=df, labels="type")
# stats = GroupComparisonAnalyzer(dataset=df, labels="type")

# dataset = preprocess.preprocessing(dataset=df, labels="type")
# res = stats.run_group_comparison()

# %%

import  os
import pandas as pd
import numpy as np


from autostats.preprocessing import Preprocess
from autostats.autostats import GroupComparisonAnalyzer



df = pd.read_csv("/media/rostyslav/Toshiba/Projects/age-related-conditions/01-data/01-raw/icr-identify-age-related-conditions/train.csv", sep=",")

preprocess = Preprocess(dataset=df, labels="Class")


dataset_prep = preprocess.preprocessing(dataset=df, labels="Class")
dataset_prep2 = dataset_prep.drop("Id", axis=1)
# dataset_prep2.remove_unused_categories()
stats = GroupComparisonAnalyzer(dataset=dataset_prep2, labels="Class", output_path="/media/rostyslav/Toshiba/Projects/autostats/02-output/")
res = stats.run_group_comparison(adjustment="bonferroni")

# %%
