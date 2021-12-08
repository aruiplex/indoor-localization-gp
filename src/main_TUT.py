import numpy as np
import pandas as pd

import utils.tut as tut
from dfio import DataFrameIO
from generate import Generate
from model import Model
from processData import ProcessData

t = tut.TUT(path="src/data/tut")
# ---------------------- <train model> ----------------------
# 3d position
xyz = t.training_data.coord_3d_scaled
# scaled WAP data
v = t.training_data.rss_scaled
# create model
m = Model(xyz, v, 992)
# ---------------------- </train model> ----------------------

# ---------------------- <generate data> ----------------------
# generate input value to prediction
generate = Generate(xyz)
# the xy input data base on floor
xyz_pred = generate.xyz_pred()
# ---------------------- </generate data> ----------------------

# geo position
xyz_pred = np.array(xyz_pred)
# wap values
v_pred = m.z_pred(xyz_pred)

processData = ProcessData()
# clean the data
v_pred_ori = processData.destandardization(v_pred)
v_pred_ori = processData.clean_pred(v_pred_ori)

# # get the dataframe header
# cols = t.training_df.columns[:992]
# # create the dataframe
# df_fake = t.training_df.df_fake(xyz_pred, v_pred_ori, cols, format="XYZ")

# # mix the real data and fake data together
# df_mix = dataFrameIO.df_mix(df, df_fake)
# # save to file
# dataFrameIO.save_df(df_mix)
