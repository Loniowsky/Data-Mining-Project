from common.outliers import drop_outliers_from_dataset
from statistics import mean, stdev
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def k_means_multiple_dim_silhouette(dataframe, columns, number_of_clusters):
    k_means_data = dataframe[columns].to_numpy()
    k_means = KMeans(n_clusters=number_of_clusters).fit(k_means_data)
    return metrics.silhouette_score(k_means_data, k_means.labels_, metric='euclidean')


def k_means_multiple_dim_calinski_harabasz(dataframe, columns, number_of_clusters):
    k_means_data = dataframe[columns].to_numpy()
    k_means = KMeans(n_clusters=number_of_clusters).fit(k_means_data)
    return metrics.calinski_harabasz_score(k_means_data, k_means.labels_)


def hierarchical_multiple_dim_silhouette(dataframe, columns, number_of_clusters):
    clustering_data = dataframe[columns].to_numpy()
    clustering = AgglomerativeClustering(n_clusters=number_of_clusters).fit(clustering_data)
    return metrics.silhouette_score(clustering_data, clustering.labels_)


def hierarchical_multiple_dim_calinski_harabasz(dataframe, columns, number_of_clusters):
    clustering_data = dataframe[columns].to_numpy()
    clustering = AgglomerativeClustering(n_clusters=number_of_clusters).fit(clustering_data)
    return metrics.calinski_harabasz_score(clustering_data, clustering.labels_)


def perform_clustering_score_analysis(dataframe, columns, numbers_of_clusters, score_strategy, outliers, repetitions):
    score_values = []
    error_values = []
    for number_of_clusters in numbers_of_clusters:
        current_number_of_clusters_scores = []
        for repetition in range(repetitions):
            data_without_outliers = drop_outliers_from_dataset(dataframe, outliers)
            silhouette_value = score_strategy(data_without_outliers, columns, number_of_clusters)
            current_number_of_clusters_scores.append(silhouette_value)
        mean_current_score = mean(current_number_of_clusters_scores)
        error_current_score = stdev(current_number_of_clusters_scores)
        print("Score for {} clusters {} (+-{})".format(number_of_clusters, mean_current_score, error_current_score))
        score_values.append(mean_current_score)
        error_values.append(error_current_score)
    return score_values, error_values


def plot_clustering_scores(numbers_of_clusters, scores, errors, method_names, score_name):
    plt.figure()
    plt.title("Clustering score: {}".format(score_name))
    plt.xlabel("Number of clusters")
    plt.ylabel("{} score value".format(score_name))
    for index, method in enumerate(method_names):
        plt.errorbar(numbers_of_clusters, scores[index], yerr=errors[index], label=method_names[index])
    plt.legend()
    plt.show()


def k_means_1d_clustering(dataframe, column, number_of_clusters):
    k_means_data = dataframe[column].to_numpy().reshape(-1, 1)
    return k_means_clustering(dataframe, k_means_data, number_of_clusters)


def k_means_multiple_dim_clustering(dataframe, columns, number_of_clusters):
    k_means_data = dataframe[columns].to_numpy()
    return k_means_clustering(dataframe, k_means_data, number_of_clusters)


def k_means_clustering(dataframe, k_means_data, number_of_clusters):
    k_means = KMeans(n_clusters=number_of_clusters).fit(k_means_data)
    return dataframe.assign(cluster=k_means.labels_)


def hierarchical_clustering(dataframe, columns, number_of_clusters):
    clustering_data = dataframe[columns].to_numpy()
    clustering = AgglomerativeClustering(n_clusters=number_of_clusters).fit(clustering_data)
    return dataframe.assign(cluster=clustering.labels_)


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


def plot_means_in_clusters_for_given_column(clustered_data, column, y_range=None, colors=None):
    ax = sns.barplot(x="cluster", y=column, data=clustered_data, ci=None, palette=colors)
    if y_range:
        ax.set(ylim=y_range)
    ax.set(xlabel="Cluster number", ylabel="Mean {}".format(column),
           title="Mean {} for each cluster".format(column))
    plt.show()


def plot_dendrogram_for_hierarchical_clustering(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            current_count += (1 if child_idx < n_samples else counts[child_idx - n_samples])
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)