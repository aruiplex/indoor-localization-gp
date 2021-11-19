import pandas as pd


f = open("./can405_indoor_localization-master/data/UJIIndoorLoc/trainingData2.csv", "r")
df = pd.read_csv(f)

f = 4

for b in range(0, 3):
    df.loc[(df['BUILDINGID'] == b) & (df["FLOOR"] == f)
           ].to_csv(f"./data_silce/building_{b}_floor_{f}.csv")
