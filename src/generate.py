import random
from loguru import logger
import numpy as np
import pandas as pd
from config import cfg
# def dataset_limitaion(df: pd.DataFrame):
#     df[]

# def increase_pred_input_nd(num, df_lim: pd.DataFrame, data_set: int):
#     """
#     generate input value to prediction, which base on uniform random.
#     The input value is LONGITUDE and LATITUDE.
#     """
#     df_lim_dataset = df_lim.loc[df_lim["DATASET"] == data_set]
#     x_min = df_lim_dataset["LONGITUDE"]["min"].values[0]
#     x_max = df_lim_dataset["LONGITUDE"]["max"].values[0]
#     y_min = df_lim_dataset["LATITUDE"]["min"].values[0]
#     y_max = df_lim_dataset["LATITUDE"]["max"].values[0]
#     l = []
#     for _ in range(num):
#         l.append(
#             [random.uniform(x_min, x_max), random.uniform(
#                 y_min, y_max), data_set]
#         )
#     xy = np.array(l)
#     return xy


class Generate:

    def boundary(self, xy):
        self.x_min = xy[:, 0].min()
        self.x_max = xy[:, 0].max()
        self.y_min = xy[:, 1].min()
        self.y_max = xy[:, 1].max()

    def xy_pred(self):
        data_set = cfg["model"]["data_set"]
        l = []
        fake_points_number = cfg["generate"]["points_num"]
        logger.info(f"fake_points_number: {fake_points_number}")
        for _ in range(fake_points_number):
            l.append([random.uniform(self.x_min, self.x_max),
                      random.uniform(self.y_min, self.y_max), data_set])
        xy_pred = np.array(l)
        return xy_pred
