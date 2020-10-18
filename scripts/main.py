from bar_plotter import BarPlotter
from data_loader import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from scatter_plotter import ScatterPlotter

data_main = DataLoader.from_file("../datasets/vg_sales_2016/vg_sales_2016.csv")
data = data_main
print(data.columns)
category_column = "Year_of_Release"
data = data[["Name", category_column, "NA_Sales", "EU_Sales", "JP_Sales", "Critic_Score", "User_Score"]]
data = data.dropna()
data = data[data.User_Score != "tbd"]
data["User_Score"] = pd.to_numeric(data["User_Score"])
print(data.count())

scatter = ScatterPlotter(data.Critic_Score, data.User_Score, data[category_column])\
    .with_labels('Critic score', 'User score')\
    .with_ticks(np.arange(0.0, 100.0, 10.0), np.arange(0.0, 10.0, 1.0))\
    .with_color_bar()\
    .plot()


data = data_main
gby = data.groupby("Genre")
means = gby.mean()[["NA_Sales", "EU_Sales", "JP_Sales"]]
print(means)

genres = gby.groups.keys()
NA_sales = means.NA_Sales.values
EU_sales = means.EU_Sales.values
JP_sales = means.JP_Sales.values

for title, sales in [("NA", NA_sales), ("EU", EU_sales), ("JP", JP_sales)]:
    BarPlotter(genres, sales)\
        .with_title(title)\
        .with_ticks_rotation(x_rotation=45).plot()


print(data_main.Platform.unique())
sony = ['PS3', 'PS2', 'PS4', 'PS', 'PSP', 'PSV']
microsoft = ["X360", "XOne", "XB"]
pc = ["PC"]
nintendo = ["Wii", "WiiU", "NES", "GB", "DS", "SNES", "GBA", "3DS", "N64", "GC"]
other = [
    "2600", # atari
    "GEN", # sega genesis
    "DC", # sega dreamcast
    "SAT", # sega saturn
    "SCD", # ???
    "WS", # ???
    "NG", # Neo Geo ???
    "TG16", # ???
    "3DO",
    "GG", # Game Gear Sega
    "PCFX"
]

sony_dict = {entry: "sony" for entry in sony}
microsoft_dict = {entry: "ms" for entry in microsoft}
nintendo_dict = {entry: "nintendo" for entry in nintendo}
other_dict = {entry: "other" for entry in other}

data_main = data_main.replace({"Platform": sony_dict})
data_main = data_main.replace({"Platform": microsoft_dict})
data_main = data_main.replace({"Platform": nintendo_dict})
data_main = data_main.replace({"Platform": other_dict})

lb_make = LabelEncoder()
category_column = "Platform"
data = data_main[[category_column, "Year_of_Release", "NA_Sales", "EU_Sales", "JP_Sales"]]
data = data.dropna()
data[category_column] = lb_make.fit_transform(data[category_column])
print(lb_make.classes_)

# TODO: year 2020 ?????
for sales in ["NA_Sales", "EU_Sales", "JP_Sales"]:
    ScatterPlotter(data.Year_of_Release, data[sales], data[category_column])\
        .with_labels("Year of release", sales)\
        .with_color_bar()\
        .plot()
