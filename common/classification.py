from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import LeaveOneOut, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn import tree
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def k_neighbours_classification(training_dataset, testing_dataset, number_of_neighbours, target_column):
    classifier = KNeighborsClassifier(n_neighbors=number_of_neighbours).fit(
        training_dataset.drop(target_column, axis=1), training_dataset[target_column])
    result = classifier.predict(testing_dataset.drop(target_column, axis=1))
    return result


def decision_tree_classification(training_dataset, testing_dataset, target_column):
    classifier = tree.DecisionTreeClassifier().fit(
        training_dataset.drop(target_column, axis=1), training_dataset[target_column])
    result = classifier.predict(testing_dataset.drop(target_column, axis=1))
    return result


def classification_leave_one_out(dataset, target_column, classifier):
    leave_one_out = LeaveOneOut()
    number_of_correct = 0
    for train_index, test_index in leave_one_out.split(dataset):
        dataset_train = dataset.iloc[train_index]
        dataset_test = dataset.iloc[test_index]
        classification_result = classifier(dataset_train, dataset_test)
        predicted_class = classification_result[0]
        actual_class = dataset_test[target_column].iloc[0]
        number_of_correct = number_of_correct + (1 if predicted_class == actual_class else 0)
    return number_of_correct / dataset.shape[0]


def k_neighbours_leave_one_out(dataset, target_column, number_of_neighbours):
    return classification_leave_one_out(dataset, target_column,
                                        lambda dataset_train, dataset_test: k_neighbours_classification(
                                            dataset_train, dataset_test, number_of_neighbours, target_column))


def decision_tree_leave_one_out(dataset, target_column):
    return classification_leave_one_out(dataset, target_column,
                                        lambda dataset_train, dataset_test: decision_tree_classification(
                                            dataset_train, dataset_test, target_column))


def decision_tree_plot_decision_surfaces(classifier, data, column_pair, palette):
    plt.figure()
    x_min, x_max = data[column_pair[0]].min() - 0.1, data[column_pair[0]].max() + 0.1
    y_min, y_max = data[column_pair[1]].min() - 0.1, data[column_pair[1]].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Greens)
    sns.scatterplot(data=data, x=column_pair[0], y=column_pair[1], hue="cluster", palette=palette)
    plt.show()


def compute_bandwidth_for_kernel_density_estimator(training_data, column_names, class_number):
    tr_data = training_data[training_data["cluster"] == class_number]
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
    grid.fit(tr_data[column_names])
    return grid.best_params_["bandwidth"]


def plot_kernel_density_estimator(training_data, column_names, class_number, bandwidth):
    tr_data = training_data[training_data["cluster"] == class_number]
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(tr_data[column_names])
    x_tmp = np.arange(-4.1, 2.5, 0.1).tolist()
    y_tmp = np.arange(-1, 2.3, 0.1).tolist()
    x_tmp, y_tmp = np.meshgrid(x_tmp, y_tmp)
    inp = np.vstack((x_tmp.ravel(), y_tmp.ravel())).T
    result = np.exp(kde.score_samples(inp))
    result = np.reshape(result, x_tmp.shape)
    fig = plt.figure()
    sns.scatterplot(data=tr_data, x="Critic_Score", y="Year_of_Release", marker='x')
    p = plt.contour(x_tmp, y_tmp, result, cmap='coolwarm')
    fig.colorbar(p, shrink=0.5, aspect=5)
    plt.show()


def perform_classification_using_kernel_density_estimator(data_to_classify, column_names, class_counts, kde_objects):
    number_of_classes = len(class_counts)
    kde_scores = []
    for class_number in range(0, number_of_classes):
        kde = kde_objects[class_number]
        kde_scores_for_class = class_counts[class_number] * np.exp(kde.score_samples(data_to_classify[column_names]))
        kde_scores.append(kde_scores_for_class)

    correct_classifications = 0
    for index, row in data_to_classify.iterrows():
        scores = [kde_scores[class_number][index] for class_number in range(0, number_of_classes)]
        max_index = scores.index(max(scores))
        if max_index == row["cluster"]:
            correct_classifications += 1

    accuracy = correct_classifications / len(data_to_classify.index)
    print("Classification accuracy: {}".format(accuracy))
    return accuracy


def plot_classification_score_for_different_parameter_values(train_accuracy, test_accuracy, parameter_range, parameter):
    plt.figure()
    plt.xlabel(parameter)
    plt.ylabel("Percent of valid classification results")
    plt.plot(parameter_range, train_accuracy, label="Score for training data")
    plt.plot(parameter_range, test_accuracy, label="Score for testing data")
    plt.legend()
    plt.show()


def plot_confusion_matrix_for_given_classifier(classifier, column_names, data_to_classify, matrix_title):
    plot_confusion_matrix(classifier, data_to_classify[column_names], data_to_classify["cluster"])
    plt.title(matrix_title)
    plt.show()


def plot_decision_tree(classifier):
    plt.figure(figsize=(15, 15))
    tree.plot_tree(classifier)
    plt.show()


def evaluate_classifier_using_accuracy_and_confusion_matrices(classifier, training_data, testing_data, column_names):

    # compute and display accuracy for both testing and training datasets
    accuracy_test = classifier.score(testing_data[column_names], testing_data["cluster"])
    print("Accuracy for testing data: {}".format(accuracy_test))
    accuracy_train = classifier.score(training_data[column_names], training_data["cluster"])
    print("Accuracy for training data: {}".format(accuracy_train))

    # display confusion matrices for both testing and training datasets
    plot_confusion_matrix_for_given_classifier(classifier, column_names, testing_data, "Testing data CM")
    plot_confusion_matrix_for_given_classifier(classifier, column_names, training_data, "Training data CM")


def decision_tree_full_classification_test(training_data, testing_data, column_names, palette):

    # perform classification for different tree depths
    accuracy_test = []
    accuracy_train = []
    for max_depth in range(1, 11):
        classifier = tree.DecisionTreeClassifier(max_depth=max_depth) \
            .fit(training_data[column_names], training_data["cluster"])

        test = classifier.score(testing_data[column_names], testing_data["cluster"])
        train = classifier.score(training_data[column_names], training_data["cluster"])

        print("Max depth = {} - test data score: {}, training data score: {}".format(max_depth, test, train))
        accuracy_test.append(test)
        accuracy_train.append(train)

    plot_classification_score_for_different_parameter_values(accuracy_train, accuracy_test,
                                                             range(1, 11), "Max depth of decision tree")

    # perform classification for max tree depth equal to 3 (avoid over-fitting)
    print("\nUsing decision tree with max depth=3 to perform in-depth tests:")
    classifier = tree.DecisionTreeClassifier(max_depth=3).fit(training_data[column_names], training_data["cluster"])
    plot_decision_tree(classifier)
    decision_tree_plot_decision_surfaces(classifier, training_data, (column_names[0], column_names[1]), palette)
    evaluate_classifier_using_accuracy_and_confusion_matrices(classifier, training_data, testing_data, column_names)


def k_neighbours_full_classification_test(training_data, testing_data, column_names, k_value=15):
    accuracy_test = []
    accuracy_train = []
    for number_of_neighbours in range(1, 31):
        classifier = KNeighborsClassifier(n_neighbors=number_of_neighbours) \
            .fit(training_data[column_names], training_data["cluster"])

        test = classifier.score(testing_data[column_names], testing_data["cluster"])
        train = classifier.score(training_data[column_names], training_data["cluster"])

        print("K = {} - test data score: {}, training data score: {}".format(number_of_neighbours, test, train))
        accuracy_test.append(test)
        accuracy_train.append(train)

    plot_classification_score_for_different_parameter_values(accuracy_train, accuracy_test,
                                                             range(1, 31), "Value of K - number of neighbours")

    # perform classification for k=15 (optimal value of k)
    print("\nUsing KNN with K={} to perform in-depth tests:".format(k_value))
    classifier = KNeighborsClassifier(n_neighbors=k_value).fit(training_data[column_names], training_data["cluster"])
    evaluate_classifier_using_accuracy_and_confusion_matrices(classifier, training_data, testing_data, column_names)


def split_dataset_into_training_and_testing_parts(data, column_names, test_size):

    # convert to numpy format
    x = data[column_names].to_numpy()
    y = data["cluster"].to_numpy()

    # split to training and testing datasets (with stratify)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y)

    # create training and testing dataframes
    training_data = pd.DataFrame(np.concatenate((x_train, np.expand_dims(y_train, axis=1)), axis=1),
                                 columns=[*column_names, "cluster"])
    training_data["cluster"] = training_data["cluster"].astype(int)
    testing_data = pd.DataFrame(np.concatenate((x_test, np.expand_dims(y_test, axis=1)), axis=1),
                                columns=[*column_names, "cluster"])
    testing_data["cluster"] = testing_data["cluster"].astype(int)

    # return tuple with training and testing datasets
    return training_data, testing_data
