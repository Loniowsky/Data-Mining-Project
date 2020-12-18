def standardize(dataframe, columns, data_stats):
    for column in columns:
        dataframe[column] = (dataframe[column] - data_stats[column]["mean"]) / data_stats[column]["std"]


def de_standardize(dataframe, columns, data_stats):
    for column in columns:
        dataframe[column] = dataframe[column] * data_stats[column]["std"] + data_stats[column]["mean"]
