import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import main_UJI


def plot_wap001():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    _file = open("./data_silce/building_0_floor_0.csv", "r")
    df: pd.DataFrame = pd.read_csv(_file)
    x = df[["LONGITUDE"]].to_numpy().T[0]
    y = df[["LATITUDE"]].to_numpy().T[0]
    z = abs(df[["WAP007"]].to_numpy().T[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    plt.show()


def plot_wap001_2():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    _file = open("./data_silce/building_0_floor_0.csv", "r")
    df: pd.DataFrame = pd.read_csv(_file)
    x = df[["LONGITUDE"]].to_numpy().T[0]
    y = df[["LATITUDE"]].to_numpy().T[0]
    z_007 = df[["WAP007"]].to_numpy().T[0]
    xx = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    x_pred, y_pred = -7632.1436, 4864982.2171
    input_pred = np.array([[x_pred, y_pred]])
    z_pred = main_UJI.predict_by_position(xx, z_007, input_pred=input_pred)[0][7]
    print(z_pred)
    # ax.plot3D(x, y, z, 'gray')
    ax.scatter(x, y, z_007, c=z_007, cmap='rainbow', label="WAP007")
    ax.scatter(x_pred, y_pred, z_pred, c=z_pred, cmap='Set1', label="pred")
    plt.legend(loc='upper left')
    plt.show()


plot_wap001_2()
