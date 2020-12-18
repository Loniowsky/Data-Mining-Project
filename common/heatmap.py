def heatmap_values(data, heat_column_name, row_name, column_name):
    results = []
    rows = data[row_name].unique()
    columns = data[column_name].unique()
    for i_idx, i in enumerate(rows):
        results.append([])
        for j in columns:
            results[i_idx].append(data[(data[row_name] == i) & (data[column_name] == j)][heat_column_name].sum())
    return [results, rows, columns]
