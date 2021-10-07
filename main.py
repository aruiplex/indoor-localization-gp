import os
import GPy
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle as pk
import sklearn.preprocessing

# def dataset_limitaion(df: pd.DataFrame):
#     df[]


def increase_pred_input_nd(num, df_lim: pd.DataFrame, data_set: int):
    """generate input value to prediction, which base on uniform random

    Args:
        num (int): the number of generating
        kwarg:  
            x_min = kwarg["x_min"]
            x_max = kwarg["x_max"]
            y_min = kwarg["y_min"]
            y_max = kwarg["y_max"]

    Returns:
        np.ndarray: a 2*num array
         df_lim.loc[df_lim["DATASET"]=="10"]

    """
    df_lim_dataset = df_lim.loc[df_lim["DATASET"] == data_set]
    x_min = df_lim_dataset["LONGITUDE"]["min"].values[0]
    x_max = df_lim_dataset["LONGITUDE"]["max"].values[0]
    y_min = df_lim_dataset["LATITUDE"]["min"].values[0]
    y_max = df_lim_dataset["LATITUDE"]["max"].values[0]
    l = []
    for _ in range(num):
        l.append(
            [random.uniform(x_min, x_max), random.uniform(
                y_min, y_max), data_set]
        )
    xy = np.array(l)
    return xy


def dump_and_load_model(m):
    filename = "./model.ml"
    if os.path.exists(filename):
        with open(filename, "rb") as _f:
            return pk.load(_f)
    else:
        with open(filename, "wb") as _f:
            m.optimize(messages=True)
            pk.dump(m, _f)
            return m


def plot(xy, z, xy_pred, z_pred, n_dim):
    """plot real data and predicted data
    all input data are n*1 shape

    Args:
        xy (np.ndarray): real data x and y coordination
        z (np.ndarray): real data rss
        xy_pred (np.ndarray): prediction x and y coordication
        z_pred (np.ndarray): prediction rss
        n_dim (int): plot the specific dimension.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot3D(x, y, z, 'gray')
    x_axis = xy[:, 0]
    y_axis = xy[:, 1]
    ax.scatter(x_axis, y_axis, z[:, n_dim], c=z[:, n_dim],
               cmap='Greys', label="WAP007")
    if xy_pred is not None:
        ax.scatter(xy_pred[:, 0], xy_pred[:, 1], z_pred[:, n_dim],
                   c=z_pred[:, n_dim], cmap='hsv', label="pred")
    plt.legend(loc='upper left')
    plt.show()


    # read file data into dataframe
with open("./can405_indoor_localization-master/data/UJIIndoorLoc/trainingData2.csv", "r") as _file:
    # all buildings incloud, except for the useless value
    # df_raw: pd.DataFrame = pd.read_csv(_file).loc[:, "LONGITUDE":"BUILDINGID"]
    df_raw = pd.read_csv(_file).loc[:, "WAP001":"BUILDINGID"]

# get mean of data which on the same position
df = df_raw.groupby(["LONGITUDE", "LATITUDE", "BUILDINGID",
                    "FLOOR"], as_index=False).mean()


# generate dataset number
df["DATASET"] = df.apply(lambda x: int(str(
    int(x["BUILDINGID"])) + str(int(x["FLOOR"]))), axis=1)

# every Building and floor have a Longitude and Latitude limitation
df_lim = df.groupby(["DATASET"], as_index=False).aggregate(
    {"LONGITUDE": ["max", "min"], "LATITUDE": ["max", "min"]})

# todo: input_dim is 2 (LONGITUDE and Latitude) or 4 (plus BuildingID and Floor)?
kernel = GPy.util.multioutput.ICM(
    input_dim=2, num_outputs=520, kernel=GPy.kern.RBF(2))


xy = df.loc[:, "LONGITUDE":"LATITUDE"].to_numpy()
z = df.loc[:, "WAP001": "WAP520"].to_numpy()
z = sklearn.preprocessing.normalize(z, norm="l2")

m = GPy.models.GPCoregionalizedRegression([xy], [z], kernel=kernel)
# m = dump_and_load_model(m)
m.optimize_restarts(num_restarts=10)
"""
data_set = 0
x_min = xy[:, 0].min()
x_max = xy[:, 0].max()
y_min = xy[:, 1].min()
y_max = xy[:, 1].max()
l = []
for _ in range(1000):
    l.append([random.uniform(x_min, x_max),
             random.uniform(y_min, y_max), data_set])
xy_pred = np.array(l)
"""
dataset = 0

xy_pred = increase_pred_input_nd(1000, df_lim, data_set=dataset)
Y_metadata = {"output_index": xy_pred[:, -1].astype(int)}
z_pred_raw = m.predict(xy_pred, Y_metadata=Y_metadata)
z_pred = z_pred_raw[0]
plot(xy, z, xy_pred, z_pred, 8)
