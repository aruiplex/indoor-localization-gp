from dfio import DataFrameIO
from processData import ProcessData
from model import Model
from generate import Generate
import numpy as np

dataFrameIO = DataFrameIO()
processData = ProcessData()

# WAP data with duplicate
df_raw = dataFrameIO.df_raw()
# WAP data without duplicate
df = processData.de_duplication(df_raw)

# ---------------------- <train model> ----------------------
# LONGITUDE and LATITUDE
xy = df.loc[:, "LONGITUDE":"LATITUDE"].to_numpy()
# raw WAP data
z_original = df.loc[:, "WAP001": "WAP520"].to_numpy()
# standardization of WAP data
z = processData.standardization(z_original)
# create model
m = Model(xy, z)
# ---------------------- </train model> ----------------------

# ---------------------- <generate data> ----------------------
group_keys = processData.get_floors_num(df)
dfs = []
for building_id, floor in group_keys:
    dfs.append(processData.split_data(df, building_id, floor))

xy_pred = []
for df_floor in dfs:
    # LONGITUDE and LATITUDE as xy
    xy_floor = df_floor.loc[:, "LONGITUDE":"LATITUDE"].to_numpy()
    # raw WAP data as z
    z_original_floor = df_floor.loc[:, "WAP001": "WAP520"].to_numpy()
    # generate input value to prediction
    generate = Generate(xy_floor)
    # the xy input data base on floor
    xy_pred_floor = generate.xy_pred()
    # add all floor input prediect data together
    xy_pred.extend(xy_pred_floor)
    # xy_pred = np.append(xy_pred, xy_pred_floor, axis=1)

# ---------------------- </generate data> ----------------------

xy_pred = np.array(xy_pred)
z_pred = m.z_pred(xy_pred)

z_pred_ori = processData.destandardization(z_pred)

z_pred_ori = processData.clean_pred(z_pred_ori)

cols = df_raw.columns[:520]
df_fake = dataFrameIO.df_fake(xy_pred, z_pred_ori, cols)

# mix the real data and fake data together 

df_mix = dataFrameIO.df_mix(df, df_fake)
dataFrameIO.save_df(df_mix)