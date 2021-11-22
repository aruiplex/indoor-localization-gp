import GPy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def read_data(filename: str):
    """read the data from csv file

    Returns:
    """
    _file = open(filename, "r")
    df: pd.DataFrame = pd.read_csv(_file)
    return df


df_raw = read_data("./data_silce/building_0_floor_0.csv")


def clean_data_nd(df: pd.DataFrame):
    """Remove the duplicate data from the dataframe

    Args:

    Returns:
        pd.Dataframe: the cleaned data
    """
    df_sub = df[["WAP007", "WAP008", "WAP009",
                 "WAP010", "LONGITUDE", "LATITUDE"]]
    df_sub.groupby(["LONGITUDE", "LATITUDE"]).size(
    ).reset_index().rename(columns={0: "count"})
    # d = {"LONGITUDE": "LONGITUDE", "LATITUDE": "LATITUDE", "WAP007": "WAP007"}
    # df = df_sub.groupby(["LONGITUDE", "LATITUDE"]).agg({"WAP007": "mean"})
    df = df_sub.groupby(["LONGITUDE", "LATITUDE"],
                        as_index=False).aggregate({"WAP007": "mean", "WAP008": "mean", "WAP009": "mean", "WAP010": "mean"})
    return df


df = clean_data_nd(df_raw)


def modeling_nd(xy: np.ndarray, z: np.ndarray):
    # kernel = GPy.kern.RBF(1,lengthscale=1, ARD = True)**GPy.kern.Coregionalize(input_dim=1,output_dim=4, rank=1)

    # kernel = GPy.util.multioutput.ICM(
    kernel = GPy.util.multioutput.ICM(
        input_dim=2, num_outputs=4, kernel=GPy.kern.RBF(2))
    print(f"kernel:\n{kernel}")
    print(f"xy.shape:{xy.shape}")
    print(f"z.shape:{z.shape}")
    m = GPy.models.GPCoregionalizedRegression(
        np.array([xy]), np.array([z]), kernel=kernel)
    print(f"model:\n{m}")
    m.optimize(messages=True)
    m.optimize_restarts(num_restarts=10)
    return m


def generate_data_nd(df):
    xy = df[["LONGITUDE", "LATITUDE"]].to_numpy().T
    z = df[["WAP007", "WAP008", "WAP009", "WAP010"]].to_numpy().T
    return xy, z


xy, z = generate_data_nd(df)
m = modeling_nd(xy.T, z.T)


def increase_pred_input_nd(num, lim):
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
    """
    x_min = lim["x_min"]
    x_max = lim["x_max"]
    y_min = lim["y_min"]
    y_max = lim["y_max"]
    l = []
    for _ in range(num):
        l.append(
            [random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0]
        )
    xy = np.array(l)
    return xy


# xy_pred = np.array([xy[0].min(), xy[1].min(), 0,
#                    xy[0].max(), xy[1].max(), 0]).reshape(-1, 3)
# xy_pred = np.array([xy[0].min(), xy[1].min(), 0,
#                    xy[0].max(), xy[1].max(), 0]).reshape(-1, 3)
limitation = {
    "x_min": xy[0].min(), "x_max": xy[0].max(),
    "y_min": xy[1].min(), "y_max": xy[1].max()
}
xy_pred = increase_pred_input_nd(1000, limitation)


xy_pred[:, -1:]


z_pred_raw = m.predict(xy_pred, Y_metadata={"output_index": xy_pred[:, -1].astype(int)})
# z_pred = m.predict(xy_pred, Y_metadata={
#                    "output_index": xy_pred[:, -1].astype(int)})
# z_pred = m.predict(xy_pred)


# z_pred = z_pred_raw[0][:,0]


def plot(xy, z, xy_pred, z_pred):
    """plot real data and predicted data

    Args:
        xy (np.ndarray): real data x and y coordination
        z (np.ndarray): real data rss
        xy_pred (np.ndarray): prediction x and y coordication
        z_pred (np.ndarray): prediction rss
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot3D(x, y, z, 'gray')
    ax.scatter(xy[0], xy[1], z, c=z, cmap='Greys', label="WAP007")
    if xy_pred is not None:
        ax.scatter(xy_pred[0], xy_pred[1], z_pred,
                   c=z_pred, cmap='hsv', label="pred")
    plt.legend(loc='upper left')
    plt.show()


# a = xy_pred[:, :-1].T
# b = z_pred[:, np.newaxis].T
# c = z[0]
# plot(xy, c, a, b)
z_pred = z_pred_raw[0][:, 3]
a = xy_pred[:, :-1].T
b = z_pred[:, np.newaxis].T
c = z[3]
plot(xy, c, None, None)
# plot(xy, c, a, b)
