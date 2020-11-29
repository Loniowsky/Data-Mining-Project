from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import math
import numpy as np


def find_outliers_in_single_column(dataframe, column, number_of_neighbours, percent_of_outliers):
    nn_data = dataframe[column].to_numpy().reshape(-1, 1)
    return nearest_neighbors(nn_data, dataframe, number_of_neighbours, percent_of_outliers)


def find_outliers_in_two_columns(dataframe, columns: (str, str), number_of_neighbours, percent_of_outliers):
    nn_data = dataframe[[columns[0], columns[1]]].to_numpy()
    return nearest_neighbors(nn_data, dataframe, number_of_neighbours, percent_of_outliers)


def nearest_neighbors(nn_data, dataframe, number_of_neighbours, percent_of_outliers):
    number_of_outlier_elements = math.floor(percent_of_outliers * len(dataframe.index))
    neighbours = NearestNeighbors(p=2, n_neighbors=number_of_neighbours, algorithm='ball_tree').fit(nn_data)
    distances = sorted(list(map(lambda t: (t[0], t[1][-1]), enumerate(neighbours.kneighbors()[0]))), reverse=True,
                       key=lambda x: x[1])
    outliers = dataframe.iloc[list(map(lambda x: x[0], distances[0:number_of_outlier_elements]))]
    non_outliers = dataframe.iloc[list(map(lambda x: x[0], distances[(number_of_outlier_elements + 1):]))]
    return outliers, non_outliers


def drop_outliers_from_dataset(dataframe, outliers):
    return dataframe[~dataframe.isin(outliers).all(1)]


def plot_1d_data_with_outliers(outliers, non_outliers, column):
    plt.figure()
    plt.plot(outliers[column], np.zeros_like(outliers[column]), 'x', c='b', label="outliers")
    plt.plot(non_outliers[column], np.zeros_like(non_outliers[column]), 'x', c='r', label="non-outliers")
    plt.xlabel(column)
    plt.yticks([])
    plt.legend()
    plt.title("Outliers detection: {}".format(column))


def pretty_print_1d_outliers(outliers, column):
    print("Outliers according to column {}".format(column))
    for index, row in outliers.iterrows():
        print("{}: {}".format(row["Name"], row[column]))
    print()
