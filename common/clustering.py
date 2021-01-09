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


def perform_clustering_score_analysis(data, columns, numbers_of_clusters, score_strategy, repetitions):
    score_values = []
    error_values = []
    for number_of_clusters in numbers_of_clusters:
        current_number_of_clusters_scores = []
        for repetition in range(repetitions):
            silhouette_value = score_strategy(data, columns, number_of_clusters)
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


def k_means_clustering(dataframe, columns, number_of_clusters):
    k_means_data = dataframe[columns].to_numpy()
    k_means = KMeans(n_clusters=number_of_clusters).fit(k_means_data)
    return dataframe.assign(cluster=k_means.labels_)


def hierarchical_clustering(dataframe, columns, number_of_clusters):
    clustering_data = dataframe[columns].to_numpy()
    clustering = AgglomerativeClustering(n_clusters=number_of_clusters).fit(clustering_data)
    return dataframe.assign(cluster=clustering.labels_)


def plot_means_in_clusters_for_given_column(clustered_data, column, y_range=None, colors=None):
    plt.figure(figsize=(7, 7))
    ax = sns.barplot(x="cluster", y=column, data=clustered_data, ci=None, palette=colors)
    if y_range:
        ax.set(ylim=y_range)
    ax.set(xlabel="Cluster number", ylabel="Mean {}".format(column),
           title="Mean {} for each cluster".format(column))
    plt.show()


def plot_clusters_scatter(clustered_data, columns, palette=None):
    plt.figure(figsize=(7, 7))
    sns.scatterplot(data=clustered_data, x=columns[0], y=columns[1], hue="cluster", marker='x', palette=palette)
    plt.show()
