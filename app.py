from autostats.autostats import AutoStat
import  os
import pandas as pd
import numpy as np

dataset = pd.read_csv("/home/rostyslav/Projects/STAT_project/01-data/heart.csv", sep=",")

stats = AutoStat()

stats.auto_stat_test(dataset=dataset, labels="output", output_dir="/home/rostyslav/Projects/STAT_project/03-results/")
