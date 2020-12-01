from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def k_means_1d_clustering(dataframe, column, number_of_clusters):
    k_means_data = dataframe[column].to_numpy().reshape(-1, 1)
    return k_means_clustering(dataframe, k_means_data, number_of_clusters)


def k_means_2d_clustering(dataframe, columns, number_of_clusters):
    k_means_data = dataframe[[columns[0], columns[1]]].to_numpy()
    return k_means_clustering(dataframe, k_means_data, number_of_clusters)


def k_means_clustering(dataframe, k_means_data, number_of_clusters):
    k_means = KMeans(n_clusters=number_of_clusters, random_state=0).fit(k_means_data)
    k_means_enum = list(map(lambda x: (x[0], x[1]), enumerate(k_means.labels_)))
    cluster_indexes = []
    for i in range(0, number_of_clusters):
        current_cluster = list(map(lambda x: x[0], filter(lambda x: x[1] == i, k_means_enum)))
        cluster_data = dataframe.iloc[current_cluster]
        cluster_indexes.append(cluster_data)
    return cluster_indexes


def plot_2d_data_with_clusters(clusters, column_pair):
    plt.figure()
    for i, cluster in enumerate(clusters):
        plt.plot(cluster[column_pair[0]], cluster[column_pair[1]], 'x', label='cluster {}'.format(i))
    plt.xlabel(column_pair[0])
    plt.ylabel(column_pair[1])
    plt.title(column_pair)
    plt.legend()
