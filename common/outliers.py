from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import math
import numpy as np


def find_outliers_in_single_column(dataframe, column, number_of_neighbours, percent_of_outliers):
    nn_data = dataframe[column].to_numpy().reshape(-1, 1)
    return nearest_neighbors(nn_data, dataframe, number_of_neighbours, percent_of_outliers)


def find_outliers_in_multiple_columns(dataframe, columns, number_of_neighbours, percent_of_outliers):
    nn_data = dataframe[columns].to_numpy()
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


def plot_hist_1d_data_with_outliers(outliers, non_outliers, column):
    plot_hist_data_with_outliers(outliers[column], np.zeros_like(outliers[column]),
                                 non_outliers[column], np.zeros_like(non_outliers[column]), (column, ""))
    plt.yticks([])
    plt.show()


def plot_2d_data_with_outliers(outliers, non_outliers, column_pair):
    plot_data_with_outliers(outliers[column_pair[0]], outliers[column_pair[1]],
                            non_outliers[column_pair[0]], non_outliers[column_pair[1]], column_pair)
    plt.show()


def plot_data_with_outliers(outliers_x, outliers_y, non_outliers_x, non_outliers_y, column_names):
    plt.figure()
    plt.plot(outliers_x, outliers_y, 'o', c='royalblue', label="outliers", ms=3)
    plt.plot(non_outliers_x, non_outliers_y, 'o', c='orange', label="non-outliers", ms=3)
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.legend()
    plt.title("Outliers detection: {}, {}".format(column_names[0], column_names[1]))


def plot_hist_data_with_outliers(outliers_x, outliers_y, non_outliers_x, non_outliers_y, column_names):
    plt.figure()
    plt.hist(outliers_x, outliers_y, 'x', c='b', label="outliers")
    plt.hist(non_outliers_x, non_outliers_y, 'x', c='r', label="non-outliers")
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.legend()
    plt.title("Outliers detection: {}, {}".format(column_names[0], column_names[1]))


def pretty_print_1d_outliers(outliers, column):
    print("Outliers according to column {}".format(column))
    for index, row in outliers.iterrows():
        print("{}: {}".format(row["Name"], row[column]))
    print()


