
import GPy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
import random


def read_data(filename: str):
    """read the data from csv file

    Returns:
        [pd.DataFrame]: [All data from the csv file]
    """
    _file = open(filename, "r")
    df: pd.DataFrame = pd.read_csv(_file)
    return df


def norm_data(l):
    """normalize the 2d array

    Args:
        l (np.ndarray): a two dimensional array

    Returns:
        np.ndarray: the normalized data
    """
    return sklearn.preprocessing.normalize(l, norm="l2")


def clean_data(df: pd.DataFrame):
    """Remove the duplicate data from the dataframe

    Args:
        df (pd.DataFrame): the mess dataframe

    Returns:
        pd.Dataframe: the cleaned data
    """
    df_sub = df[["WAP007", "LONGITUDE", "LATITUDE"]]
    df_sub.groupby(["LONGITUDE", "LATITUDE"]).size(
    ).reset_index().rename(columns={0: "count"})
    # d = {"LONGITUDE": "LONGITUDE", "LATITUDE": "LATITUDE", "WAP007": "WAP007"}
    # df = df_sub.groupby(["LONGITUDE", "LATITUDE"]).agg({"WAP007": "mean"})
    df = df_sub.groupby(["LONGITUDE", "LATITUDE"],
                        as_index=False).aggregate({"WAP007": "mean"})
    return df


def clean_data_nd(df: pd.DataFrame):
    """Remove the duplicate data from the dataframe

    Args:
        df (pd.DataFrame): the mess dataframe

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


def modeling(xy: np.ndarray, z: np.ndarray):
    # K=GPy.kern.RBF(1)
    # B = GPy.kern.Coregionalize(input_dim=2,output_dim=2)
    # kernel = K.prod(B,name='B.K')
    # kernel = GPy.util.multioutput.ICM(
    #     input_dim=2, num_outputs=2, kernel=GPy.kern.RBF(1))
    # 1d kernel
    kernel = GPy.kern.RBF(input_dim=2)
    m = GPy.models.GPRegression(xy, z, kernel, normalizer=True)
    m.optimize(messages=True)
    m.optimize_restarts(num_restarts=10)
    return m


def modeling_nd(xy: np.ndarray, z: np.ndarray):
    # kernel = GPy.kern.RBF(1,lengthscale=1, ARD = True)**GPy.kern.Coregionalize(input_dim=1,output_dim=4, rank=1)

    kernel = GPy.util.multioutput.ICM(
        input_dim=2, num_outputs=4, kernel=GPy.kern.Matern32(2))
    # kernel = GPy.util.multioutput.ICM(
    #     input_dim=2, num_outputs=4, kernel=GPy.kern.RBF(2))
    print(f"kernel:\n{kernel}")
    print(f"xy.shape:{xy.shape}")
    print(f"z.shape:{z.shape}")
    m = GPy.models.GPCoregionalizedRegression(
        np.array([xy]), np.array([z]), kernel=kernel)
    print(f"model:\n{m}")
    m.optimize(messages=True)
    m.optimize_restarts(num_restarts=10)
    return m


def predict(m, xy_pred: np.ndarray):
    """give the prediction (z) by input value of xy

    Args:
        m (Gaussian process Model)
        xy_pred (np.ndarray): an n*2 ndarray

    Returns:
        [np.ndarray]: a n*1 ndarray
    """
    z_pred = m.predict(xy_pred)[0]
    return z_pred


def increase_pred_input(num, lim):
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
        l.append([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])
    return np.array(l).T


def increase_pred_input_nd(num, lim, output_num):
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
    dtype = [('x', 'float'), ('y', 'float'), ('output_num', 'int')]
    for _ in range(num):
        l.append(
            [random.uniform(x_min, x_max), random.uniform(
                y_min, y_max), output_num]
        )
    xy = np.array(l, dtype=dtype)
    return xy


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
    ax.scatter(xy_pred[0], xy_pred[1], z_pred,
               c=z_pred, cmap='hsv', label="pred")
    plt.legend(loc='upper left')
    plt.show()


def generate_data(df):
    xy = df[["LONGITUDE", "LATITUDE"]].to_numpy().T
    z = df["WAP007"].to_numpy().reshape(-1, 1).T
    return xy, z


def generate_data_nd(df):
    xy = df[["LONGITUDE", "LATITUDE"]].to_numpy().T
    z = df[["WAP007", "WAP008", "WAP009", "WAP010"]].to_numpy().T
    return xy, z


def main():
    df_raw = read_data("./data_silce/building_0_floor_0.csv")
    df = clean_data(df_raw)
    # transform a n*1 2d array to a 1*n 1d array
    xy, z = generate_data(df)
    # xy = norm_data(xy_raw)
    # z = norm_data(z_raw.reshape(1, -1))
    m = modeling(xy.T, z.T)
    limitation = {
        "x_min": xy[0].min(), "x_max": xy[0].max(),
        "y_min": xy[1].min(), "y_max": xy[1].max()
    }
    xy_pred = increase_pred_input(1000, limitation)
    z_pred = predict(m, xy_pred.T)
    plot(xy, z, xy_pred, z_pred.T[0])


def main_nd():
    df_raw = read_data("./data_silce/building_0_floor_0.csv")
    df = clean_data_nd(df_raw)
    # transform a n*1 2d array to a 1*n 1d array
    xy, z = generate_data_nd(df)
    print(f"input dimension: {xy.ndim}")
    print(f"output dimension: {z.ndim}")
    m = modeling_nd(xy.T, z.T)
    limitation = {
        "x_min": xy[0].min(), "x_max": xy[0].max(),
        "y_min": xy[1].min(), "y_max": xy[1].max()
    }
    # xy_pred = increase_pred_input_nd(1000, limitation, 1)
    xy_pred = np.array([xy[0].min(), xy[1].min(), 0])[:, np.newaxis]

    # print(f"xy_pred.shape: {xy_pred.shape}")
    z_pred = predict(m, xy_pred)
    plot(xy, z, xy_pred, z_pred)


if __name__ == "__main__":
    # main_nd()
    main()
