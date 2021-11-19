import pandas as pd


def read_data():
    _file = open("./data_silce/building_0_floor_0.csv", "r")
    df: pd.DataFrame = pd.read_csv(_file)
    xy = df[["LONGITUDE", "LATITUDE"]].to_numpy()
    z = df[["WAP007"]].to_numpy()[1:521]
    # y = df.to_numpy() # WAP001 - WPA520
    return xy, z
