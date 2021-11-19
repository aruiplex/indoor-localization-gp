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

        df = df_raw.groupby(["LONGITUDE", "LATITUDE"], as_index=False).mean()
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
