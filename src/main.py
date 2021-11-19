from dfio import DataFrameIO
from processData import ProcessData
from model import Model
from generate import Generate

dataFrameIO = DataFrameIO()
processData = ProcessData()
generate = Generate()

# WAP data with duplicate
df_raw = dataFrameIO.df_raw()
# WAP data without duplicate
df = processData.de_duplication(df_raw)

# LONGITUDE and LATITUDE
xy = df.loc[:, "LONGITUDE":"LATITUDE"].to_numpy()
# raw WAP data
z_original = df.loc[:, "WAP001": "WAP520"].to_numpy()
# standardization of WAP data
z = processData.standardization(z_original)

xy_pred = generate.xy_pred()

m = Model(xy, z)
z_pred = m.z_pred(xy_pred)

z_pred_ori = processData.destandardization(z_pred)
cols = df_raw.columns[:520]
df_new = dataFrameIO.df_new(xy_pred, z_pred_ori, cols)
