import numpy as np
import tut
from dfio import DataFrameIO
from generate import Generate
from model import Model
from processData import ProcessData

t = tut.TUT(path="src/data/tut")
# 3d, xy is continuous and z is discrete
xyz = t.training_data.coord_3d_scaled
xy = xyz[:, 0:2]
z = xyz[:, -1]
# scaled WAP data
rss = t.training_data.rss_scaled

# scaler
scaler_coord_3d = t.training_data.coord_3d_scaler
scaler_rss = t.training_data.rss_scaler
# ---------------------- <train model> ----------------------

# create model
m = Model(xyz, rss, 992)
# ---------------------- </train model> ----------------------

# ---------------------- <generate data> ----------------------
# generate input value to prediction
generate = Generate(xy)
# the xy input data base on floor
xyz_pred = generate.xyz_pred()
# ---------------------- </generate data> ----------------------

# geo position
xyz_pred = np.array(xyz_pred)
# wap values
v_pred = m.rss_pred(xyz_pred)

cleaner = ProcessData()
# clean the data
v_pred_ori = scaler_rss.inverse_transform(v_pred)
v_pred_ori = cleaner.clean_pred(v_pred_ori)
# # get the dataframe header
# cols = t.training_df.columns[:992]
# # create the dataframe
# df_fake = t.training_df.df_fake(xyz_pred, v_pred_ori, cols, format="XYZ")

# # mix the real data and fake data together
# df_mix = dataFrameIO.df_mix(df, df_fake)
# # save to file
# dataFrameIO.save_df(df_mix)
