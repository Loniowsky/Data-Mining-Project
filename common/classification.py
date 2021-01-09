from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
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
