from loguru import logger
import pandas as pd
from config import cfg


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

    def df_fake(self, xy_pred, z_pred_ori, columns):
        df_new = pd.DataFrame(z_pred_ori, columns=columns).astype("int64")
        df_new["LONGITUDE"] = xy_pred[:, 0]
        df_new["LATITUDE"] = xy_pred[:, 1]
        # df_new["BUILDINGID"] = 0
        # df_new["FLOOR"] = 0
        # df_new["SPACEID"] = 0
        # df_new["RELATIVEPOSITION"] = 0
        return df_new

    def df_mix(self, df_ori, df_new):
        df_new = df_new.append(df_ori)
        df_new.reset_index()
        logger.info(f"df_mix generate finish.")
        return df_new

    def save_df(self, df_mix:pd.DataFrame):
        with open(cfg["file"]["train"], "w") as f:
            f.write(df_mix.to_csv(index=False))
            logger.info(f'training data write finsh, to {cfg["file"]["train"]}.')
