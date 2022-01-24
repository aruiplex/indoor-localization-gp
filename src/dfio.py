from loguru import logger
import pandas as pd
from config import cfg
import numpy as np


class DataFrameIO:
    """
    read file data into dataframe
    """

    def _read_file(self, file_name):
        data = pd.read_csv(file_name, sep=',')
        return data

    def df_raw(self):
        """return the raw dataframe of the file in config 

        Returns:
            pd.dataframe: all buildings incloud, except for the useless value
        """
        with open(cfg["file"]["df_file"], "r") as _file:
            # df_raw: pd.DataFrame = pd.read_csv(_file).loc[:, "LONGITUDE":"BUILDINGID"]
            df_raw = pd.read_csv(_file).loc[:, "WAP001":"BUILDINGID"]
        return df_raw

    def df_raw_tut(self):
        """return the raw dataframe of the file in config 

        Returns:
            pd.dataframe: all buildings incloud, except for the useless value
        """
        with open(cfg["file"]["df_file"], "r") as _file:
            # drop unname:0 and refpoint
            df_raw = pd.read_csv(_file).iloc[:, 1:-1]
        return df_raw

    def df_fake(self, position_pred, rss_pred_ori, columns, format="LL"):
        """Companion the fake xy and z into dataframe.

        Args:
            xy_pred (numpy array): 2d numpy array, LONGITUDE and LATITUDE.
            z_pred_ori (numpy array): wap data.
            columns (dataframe index): dataframe header

        Returns:
            dataframe: fake dataframe
        """
        # def floor_chooser(x):

        df_new = pd.DataFrame(rss_pred_ori, columns=columns).astype("int64")
        if format == "LL":
            df_new["LONGITUDE"] = position_pred[:, 0]
            df_new["LATITUDE"] = position_pred[:, 1]
        else:
            df_new["X"] = list(
                map(lambda x: str(x), position_pred[:, 0].round(3)))
            df_new["Y"] = list(
                map(lambda x: str(x), position_pred[:, 1].round(3)))
            df_new["Z"] = position_pred[:, 2]
            df_new["FLOOR"] = [{
                0: 0,
                3.7: 1,
                7.4: 2,
                11.1: 3,
                14.8: 4
            }[x] for x in position_pred[:, 2]]
            df_new["FLOOR"] = df_new["FLOOR"].astype(int)
            df_new["BUILDINGID"] = 0
            df_new["BUILDINGID"] = df_new["BUILDINGID"].astype(int)

            df_new["REFPOINT"] = 0
            df_new["REFPOINT"] = df_new["REFPOINT"].astype(int)
        return df_new

    def df_mix(self, df_ori, df_new):
        """Mix the real dataframe and fake dataframe together. 

        Args:
            df_ori (dataframe): real data
            df_new (dataframe): fake data

        Returns:
            dataframe: the mixed dataframe
        """
        df_new = df_new.append(df_ori)
        df_new.reset_index()
        logger.info(f"df_mix generate finish.")
        return df_new

    def save_df(self, df_mix: pd.DataFrame):
        """Save the dataframe into csv file.
        """
        df_mix.to_csv(cfg["file"]["train"], index=False, line_terminator='\n')
        logger.info(
            f'training data write finsh, to {cfg["file"]["train"]}.')
