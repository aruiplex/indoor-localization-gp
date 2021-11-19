import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import GPy


class DataGenerater:
    def generate_data_linear(self):
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # data = np.empty(shape=(1, 11))
        for row in range(1, 20):
            data_i = np.arange(row, row+11)
            data = np.vstack((data, data_i))
        return data

    def generate_data_two_columns_curve(self):
        """
        [
            [  1   1]
            [  2   4]
            [  3   9]
            [  4  16]
            [  5  25]
            [  6  36]
            [  7  49]
            [  8  64]
            [  9  81]
            [ 10 100]
            [ 11 121]
            [ 12 144]
            [ 13 169]
            [ 14 196]
            [ 15 225]
            [ 16 256]
            [ 17 289]
            [ 18 324]
            [ 19 361]
            [ 20 400]
        ]
        Returns:
            [type]: [description]
        """
        arr = np.arange(1, 21).reshape(-1, 1)
        arr = np.append(arr, (np.arange(1, 21)**2).reshape(-1, 1), axis=1)
        return arr

    def generate_data_real_two_column(self):
        _file = open("../data_silce/building_0_floor_0.csv", "r")
        df: DataFrame = pd.read_csv(_file)
        sub_df = df[["WAP007", "WAP008"]]
        return sub_df.to_numpy()

# def generate_data_curve():
#     data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     # data = np.empty(shape=(1, 11))
#     for row in range(1, 10):
#         for item in range(1,10):

#         data_i = np.arange(row, row+11)
#         data = np.vstack((data, data_i))


dg = DataGenerater()
# data = generate_data_linear()
data = dg.generate_data_two_columns_curve()
print("data is:")
print(data)

data_arr = data
x = np.arange(len(data_arr)).reshape((-1, 1))
y = data_arr
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(x, y, kernel)
m.optimize(messages=True)

x_pred = np.array([[100]])
y_pred = m.predict(x_pred)[0]
m.plot()
plt.show()

print(y_pred)
