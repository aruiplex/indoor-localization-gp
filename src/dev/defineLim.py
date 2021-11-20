"""
The python file is used to findout the langitude and latitude range of the dataset.
"""


import pandas as pd

# add python path to find src code.
import sys
sys.path.append("/home/aruix/aruixDAO/Code/gaussian_process/src")

df = pd.read_csv("/home/aruix/aruixDAO/Code/gaussian_process/data/campus_clean.csv").filter(
    ["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"]).copy()
print(df.groupby(["BUILDINGID", "FLOOR"]).max())
print(df.groupby(["BUILDINGID", "FLOOR"]).min())

output = """
                    LONGITUDE      LATITUDE
BUILDINGID FLOOR                           
0          0     -7587.041400  4.865012e+06
           1     -7587.102300  4.865017e+06
           2     -7587.102300  4.865017e+06
           3     -7587.102300  4.865017e+06
1          0     -7404.491683  4.864952e+06
           1     -7408.714526  4.864960e+06
           2     -7408.695251  4.864959e+06
           3     -7411.625445  4.864952e+06
2          0     -7300.818990  4.864860e+06
           1     -7303.794296  4.864862e+06
           2     -7308.779112  4.864860e+06
           3     -7309.517500  4.864862e+06
           4     -7309.517500  4.864853e+06
"""
