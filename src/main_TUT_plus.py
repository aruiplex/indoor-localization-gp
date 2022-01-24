from dfio import DataFrameIO
from processData import ProcessData
from model import Model
from generate import Generate
import numpy as np

dataFrameIO = DataFrameIO()
processData = ProcessData()

# WAP data with duplicate
df = dataFrameIO.df_raw_tut()
# WAP data without duplicate
# df = processData.de_duplication(df_raw)

# ---------------------- <train model> ----------------------
# LONGITUDE and LATITUDE
xyz = df.loc[:, "X":"Z"].to_numpy()
# raw WAP data
rss_original = df.iloc[:, :-5].to_numpy()
# standardization of WAP data
rss = processData.standardization(rss_original)
output_dim = xyz.shape[1]
# create model
m = Model(xyz, rss, output_dim)
# ---------------------- </train model> ----------------------

# ---------------------- <generate data> ----------------------
group_keys = processData.get_floors_num(df)
df_floors = []
for building_id, floor in group_keys:
    df_floors.append(processData.split_data(df, building_id, floor))

xyz_pred = []
floor = 0
for df_floor in df_floors:
    # LONGITUDE and LATITUDE as xy
    xyz_floor = df_floor.loc[:, "X":"Z"].to_numpy()
    # raw WAP data as z
    # z_original_floor = df_floor.iloc[:, 0: 992].to_numpy()
    # generate input value to prediction
    generate = Generate(xyz_floor)
    # the xy input data base on floor
    xyz_pred_floor = generate.position_pred()
    # add all floor input prediect data together
    xyz_pred.extend(xyz_pred_floor)
    # xy_pred = np.append(xy_pred, xy_pred_floor, axis=1)

# ---------------------- </generate data> ----------------------

# geo position
xyz_pred = np.array(xyz_pred)
# wap values
rss_pred = m.rss_pred(xyz_pred)

# clean the data
rss_pred = processData.destandardization(rss_pred)
rss_pred = processData.clean_pred(rss_pred)

# get the dataframe header
cols = df.columns[:992]
# create the dataframe
df_fake = dataFrameIO.df_fake(xyz_pred, rss_pred, cols, format="tut")

# mix the real data and fake data together
df_mix = dataFrameIO.df_mix(df, df_fake)
# save to file
dataFrameIO.save_df(df_mix)
