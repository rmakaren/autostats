from autostats.autostats import AutoStat
import  os
os.chdir("/media/rostyslav/Toshiba/Projects/autostats/02-scripts/autostats/")

import pandas as pd
import numpy as np

dataset = pd.read_csv("/media/rostyslav/Toshiba/Projects/autostats/01-data/heart.csv", sep=",")

stats = AutoStat()

stats.auto_stat_test(dataset=dataset, labels="output", output_dir="/media/rostyslav/Toshiba/Projects/autostats/02-scripts/autostats/output")
