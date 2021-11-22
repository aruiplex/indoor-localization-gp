import pandas as pd
import sklearn.preprocessing
from loguru import logger


class ProcessData:
    def de_duplication(self, df_raw: pd.DataFrame):
        """get mean of data which on the same position to reduce repeated data

        Args:
            df_raw (pd.dataframe): the data which need to be de-duplicated. 
            The raw dataframe has same position means both longitude and latitude are same.

        Returns:
            pd.dataframe: duplicated-free dataframe
        """

        df = df_raw.groupby(
            ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"], as_index=False).mean()
        logger.info("De-duplication done.")
        return df

    def standardization(self, z_original):
        self.standarder = sklearn.preprocessing.StandardScaler()
        self.standarder.fit(z_original)
        z = self.standarder.transform(z_original)
        logger.info("Standardization done.")
        return z

    def destandardization(self, z_pred):
        z_pred_ori = self.standarder.inverse_transform(z_pred).round()
        logger.info("Destandardization done.")
        return z_pred_ori

    def get_floors_num(self, df):
        group_keys = df.groupby(["BUILDINGID", "FLOOR"]).groups.keys()
        logger.info(f"get all floors num, floors num: {len(group_keys)}")
        return list(group_keys)

    def split_data(self, df, building_id, floor):
        logger.info(f"split data, building_id: {building_id}, floor: {floor}")
        return df[(df["BUILDINGID"] == building_id) & (df["FLOOR"] == floor)]

    def clean_pred(self, z_pred):
        for z_row in z_pred:
            for z in z_row:
                if z < -110:
                    z = -110
        return z_pred
