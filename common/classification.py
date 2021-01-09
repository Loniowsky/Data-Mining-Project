from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree


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
