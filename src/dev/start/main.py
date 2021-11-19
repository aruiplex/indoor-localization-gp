import os
import pickle
import GPy
import numpy as np
import pandas as pd


def handle_b_0_f0() -> np.ndarray:
    _file = open("./data_silce/building_0_floor_0.csv", "r")
    df: pd.DataFrame = pd.read_csv(_file)
    # arr = df.to_numpy()
    return df


def generate_multi_x_y():
    _file = open("./data_silce/building_0_floor_0.csv", "r")
    df: pd.DataFrame = pd.read_csv(_file)
    x = df[["LONGITUDE", "LATITUDE"]].to_numpy()
    y = df[["WAP007"]].to_numpy()
    # y = df.to_numpy() # WAP001 - WPA520
    return x, y


def handler_all_data():
    """check which column has not -110 data
    """
    arr = handle_b_0_f0().to_numpy()
    for i in range(1, 521):
        rss_max = arr[:, i].max()
        if int(rss_max) != -110:
            print(f"i: {i}, rss_max = {rss_max}")


def write_into_file(l):
    file_ = open("./temp", "w")
    file_.write(l)


def handle_7_column():
    """use gpy to regression 
    """
    df = handle_b_0_f0()
    # sub_df = df[["WAP007", "SPACEID"]]
    sub_df = df["WAP007"]
    arr: np.ndarray = sub_df.to_numpy()
    arr = np.sort(arr, axis=0)
    x = np.arange(len(arr)).reshape((-1, 1))
    y = arr[:, None]
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(x, y, kernel)
    m.optimize(messages=True)
    m.plot()
    plt.show()


def handle_multi_columns():
    """use gpy to regression 
    """
    df = handle_b_0_f0()
    arr: np.ndarray = df.to_numpy()
    x = np.arange(len(arr)).reshape((-1, 1))
    y = arr
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(x, y, kernel)
    m.optimize(messages=True)
    pred = m.predict(np.array([[1061]]))
    value: np.ndarray = pred[0].round().astype(int)
    l = value.tolist()
    write_into_file(str(l))
    # print(pred)
    # m.plot()
    # plt.show()


def predict_by_position(input, output, input_pred):
    filename = "./model_007_008"
    if not os.path.isfile(filename):
        f = open(filename, "wb")
        kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(input, output, kernel)
        m.optimize(messages=True)
        pickle.dump(m, f)
    else:
        f = open(filename, "rb")
        m = pickle.load(f)
    #
    output_pred = m.predict(input_pred)[0].round().astype(int)
    return output_pred


def model_train(input, output, input_pred):
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(input, output, kernel)
    m.optimize(messages=True)
    output_pred = m.predict(input_pred)[0].round().astype(int)
    return output_pred





if __name__ == "__main__":
    # x, y = generate_multi_x_y()
    # print("x: ")
    # print(x)
    # print("y: ")
    # print(y)
    # predict_by_position(x, y)
    plot_wap001()
