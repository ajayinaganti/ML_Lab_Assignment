import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def manual_dot(vec_x, vec_y):
    dot_val = 0
    for i in range(len(vec_x)):
        dot_val += vec_x[i] * vec_y[i]
    return dot_val

def euclidean_norm(vector):
    sum_sq = 0
    for element in vector:
        sum_sq += element * element
    return math.sqrt(sum_sq)

def euclidean_distance(vec1, vec2):
    dist_sum = 0
    for i in range(len(vec1)):
        dist_sum += (vec1[i] - vec2[i]) ** 2
    return math.sqrt(dist_sum)

def manual_mean(data_list):
    total_val = 0
    for item in data_list:
        total_val += item
    return total_val / len(data_list)

def manual_variance(data_list):
    avg = manual_mean(data_list)
    var_sum = 0
    for item in data_list:
        var_sum += (item - avg) ** 2
    return var_sum / len(data_list)

def manual_std(data_list):
    return math.sqrt(manual_variance(data_list))

def mean_vector(data_matrix):
    avg_vector = []
    for col_index in range(len(data_matrix[0])):
        col_values = []
        for row_index in range(len(data_matrix)):
            col_values.append(data_matrix[row_index][col_index])
        avg_vector.append(manual_mean(col_values))
    return avg_vector

def minkowski_distance(vec1, vec2, power):
    total = 0
    for i in range(len(vec1)):
        total += abs(vec1[i] - vec2[i]) ** power
    return total ** (1 / power)


def train_test_split_manual(features, labels, ratio):
    cut_point = int(len(features) * (1 - ratio))
    return (
        features[:cut_point],
        features[cut_point:],
        labels[:cut_point],
        labels[cut_point:]
    )


def knn_train(train_features, train_labels, neighbors):
    return {"X": train_features, "y": train_labels, "k": neighbors}

def knn_predict(model_data, query_point):
    distance_list = []
    for i in range(len(model_data["X"])):
        dist = euclidean_distance(model_data["X"][i], query_point)
        distance_list.append((dist, model_data["y"][i]))

    distance_list.sort(key=lambda x: x[0])

    vote_counter = {}
    for i in range(model_data["k"]):
        class_label = distance_list[i][1]
        vote_counter[class_label] = vote_counter.get(class_label, 0) + 1

    chosen_label = None
    highest_votes = -1
    for label in vote_counter:
        if vote_counter[label] > highest_votes:
            highest_votes = vote_counter[label]
            chosen_label = label

    return chosen_label

def knn_predict_all(model_data, test_features):
    prediction_list = []
    for point in test_features:
        prediction_list.append(knn_predict(model_data, point))
    return prediction_list

def knn_accuracy_manual(true_labels, predicted_labels):
    correct_count = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct_count += 1
    return correct_count / len(true_labels)

def manual_confusion_matrix(actual, predicted):
    true_pos = false_pos = true_neg = false_neg = 0
    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            true_pos += 1
        elif actual[i] == 0 and predicted[i] == 0:
            true_neg += 1
        elif actual[i] == 0 and predicted[i] == 1:
            false_pos += 1
        elif actual[i] == 1 and predicted[i] == 0:
            false_neg += 1
    return true_pos, false_pos, true_neg, false_neg

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(p_val, r_val):
    return 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) != 0 else 0

def main():
    data_points = [
        [1, 2],
        [2, 3],
        [3, 4],
        [6, 5],
        [7, 8],
        [8, 7]
    ]
    class_labels = [0, 0, 0, 1, 1, 1]

    vec_a = [1, 2, 3]
    vec_b = [4, 5, 6]
    print("Manual Dot:", manual_dot(vec_a, vec_b))
    print("Manual Norm A:", euclidean_norm(vec_a))
    print("Manual Norm B:", euclidean_norm(vec_b))
    print("NumPy Dot:", np.dot(vec_a, vec_b))
    print("NumPy Norm A:", np.linalg.norm(vec_a))

    group_zero = data_points[:3]
    group_one = data_points[3:]
    mean_zero = mean_vector(group_zero)
    mean_one = mean_vector(group_one)
    print("Class 0 Mean:", mean_zero)
    print("Class 1 Mean:", mean_one)
    print("Inter-class Distance:", euclidean_distance(mean_zero, mean_one))

    feature_values = [row[0] for row in data_points]
    print("Feature Mean:", manual_mean(feature_values))
    print("Feature Variance:", manual_variance(feature_values))
    plt.hist(feature_values, bins=5)
    plt.title("Feature Histogram")
    plt.show()

    minkowski_vals = []
    for p in range(1, 11):
        minkowski_vals.append(minkowski_distance(data_points[0], data_points[1], p))
    print("Minkowski Distances:", minkowski_vals)
    plt.plot(range(1, 11), minkowski_vals)
    plt.xlabel("p value")
    plt.ylabel("Distance")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split_manual(
        data_points, class_labels, 0.33
    )
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print()

    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    sklearn_knn.fit(X_train, y_train)
    print("Sklearn Accuracy:", sklearn_knn.score(X_test, y_test))
    sklearn_predictions = sklearn_knn.predict(X_test)
    print("Sklearn Predictions:", sklearn_predictions)
    print()

    manual_knn = knn_train(X_train, y_train, 3)
    manual_predictions = knn_predict_all(manual_knn, X_test)
    manual_accuracy = knn_accuracy_manual(y_test, manual_predictions)
    print("Manual kNN Accuracy:", manual_accuracy)
    print()

    k_range = range(1, 12)
    accuracy_curve = []
    for k in k_range:
        temp_model = knn_train(X_train, y_train, k)
        temp_preds = knn_predict_all(temp_model, X_test)
        accuracy_curve.append(knn_accuracy_manual(y_test, temp_preds))

    plt.plot(k_range, accuracy_curve, marker='o')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k")
    plt.show()

    tp, fp, tn, fn = manual_confusion_matrix(y_test, manual_predictions)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)

    print("Confusion Matrix:")
    print("TP:", tp, "FP:", fp)
    print("FN:", fn, "TN:", tn)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

main()
