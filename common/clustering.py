from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


def k_means_multiple_dim_silhouette(dataframe, columns, number_of_clusters):
    k_means_data = dataframe[columns].to_numpy()
    k_means = KMeans(n_clusters=number_of_clusters).fit(k_means_data)
    return metrics.silhouette_score(k_means_data, k_means.labels_, metric='euclidean')


def k_means_1d_clustering(dataframe, column, number_of_clusters):
    k_means_data = dataframe[column].to_numpy().reshape(-1, 1)
    return k_means_clustering(dataframe, k_means_data, number_of_clusters)


def k_means_multiple_dim_clustering(dataframe, columns, number_of_clusters):
    k_means_data = dataframe[columns].to_numpy()
    return k_means_clustering(dataframe, k_means_data, number_of_clusters)


def k_means_clustering(dataframe, k_means_data, number_of_clusters):
    k_means = KMeans(n_clusters=number_of_clusters).fit(k_means_data)
    return dataframe.assign(cluster=k_means.labels_)


def plot_1d_data_with_clusters(clustered_data, column_name):
    plt.figure()
    number_of_clusters = clustered_data["cluster"].max() + 1
    for cluster_index in range(0, number_of_clusters):
        cluster_data = clustered_data[clustered_data["cluster"] == cluster_index]
        plt.plot(cluster_data[column_name], np.zeros_like(cluster_data[column_name]), 'x',
                 label='cluster {}'.format(cluster_index))
    plt.xlabel(column_name)
    plt.ylabel("")
    plt.yticks([])
    plt.title(column_name)
    plt.legend()
    plt.show()


def plot_2d_data_with_clusters(clustered_data, column_pair):
    plt.figure()
    number_of_clusters = clustered_data["cluster"].max() + 1
    for cluster_index in range(0, number_of_clusters):
        cluster_data = clustered_data[clustered_data["cluster"] == cluster_index]
        plt.plot(cluster_data[column_pair[0]], cluster_data[column_pair[1]], 'x',
                 label='cluster {}'.format(cluster_index))
    plt.xlabel(column_pair[0])
    plt.ylabel(column_pair[1])
    plt.title(column_pair)
    plt.legend()
    plt.show()


def plot_means_in_clusters_for_given_column(clustered_data, means_table, column):
    plt.figure()
    number_of_clusters = clustered_data["cluster"].max() + 1
    plt.bar(means_table.index, means_table[column])
    plt.title("Mean {} for each cluster".format(column))
    plt.xlabel("Cluster number")
    plt.xticks(range(0, number_of_clusters))
    plt.ylabel("Mean {}".format(column))
    plt.show()
