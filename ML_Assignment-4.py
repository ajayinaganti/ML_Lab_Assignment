import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)


def knn_train_and_predict(train_features, test_features, train_labels, test_labels, neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=neighbors)
    knn_model.fit(train_features, train_labels)
    train_predictions = knn_model.predict(train_features)
    test_predictions = knn_model.predict(test_features)
    return train_predictions, test_predictions

def get_classification_scores(actual_labels, predicted_labels):
    matrix = confusion_matrix(actual_labels, predicted_labels)
    prec = precision_score(actual_labels, predicted_labels)
    rec = recall_score(actual_labels, predicted_labels)
    f1_val = f1_score(actual_labels, predicted_labels)
    return matrix, prec, rec, f1_val


def calculate_regression_scores(true_values, estimated_values):
    mse_val = mean_squared_error(true_values, estimated_values)
    rmse_val = np.sqrt(mse_val)
    mape_val = np.mean(np.abs((true_values - estimated_values) / true_values)) * 100
    r2_val = r2_score(true_values, estimated_values)
    return mse_val, rmse_val, mape_val, r2_val


def create_synthetic_training_data():
    np.random.seed(10)
    feature_matrix = np.random.randint(1, 11, size=(20, 2))
    class_labels = np.array([0 if (row[0] + row[1]) < 12 else 1 for row in feature_matrix])
    return feature_matrix, class_labels

def visualize_training_data(points, labels):
    point_colors = ['blue' if lbl == 0 else 'red' for lbl in labels]
    plt.scatter(points[:, 0], points[:, 1], c=point_colors)
    plt.xlabel("X Feature")
    plt.ylabel("Y Feature")
    plt.title("Training Data Scatter Plot")
    plt.show()


def create_dense_test_grid():
    x_axis = np.arange(0, 10, 0.1)
    y_axis = np.arange(0, 10, 0.1)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    combined_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    return combined_points

def knn_classify_and_visualize(train_points, train_classes, grid_points, neighbors):
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(train_points, train_classes)
    grid_predictions = classifier.predict(grid_points)
    plot_colors = ['blue' if val == 0 else 'red' for val in grid_predictions]
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=plot_colors, s=1)
    plt.title(f"kNN Classification (k = {neighbors})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def read_parkinsons_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    input_features = dataset[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
    target_labels = dataset['status']
    return input_features, target_labels


def tune_knn_hyperparameter(train_inputs, train_outputs):
    search_space = {'n_neighbors': np.arange(1, 21)}
    base_model = KNeighborsClassifier()
    grid_search = GridSearchCV(base_model, search_space, cv=5)
    grid_search.fit(train_inputs, train_outputs)
    return grid_search.best_params_, grid_search.best_score_


features, labels = read_parkinsons_dataset("parkinsons (1).csv")

X_tr, X_te, y_tr, y_te = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

train_out, test_out = knn_train_and_predict(X_tr, X_te, y_tr, y_te, neighbors=3)

train_matrix, train_precision, train_recall, train_f1 = get_classification_scores(
    y_tr, train_out
)
test_matrix, test_precision, test_recall, test_f1 = get_classification_scores(
    y_te, test_out
)

print("Training Confusion Matrix:\n", train_matrix)
print("Training Precision:", train_precision)
print("Training Recall:", train_recall)
print("Training F1 Score:", train_f1)

print("\nTest Confusion Matrix:\n", test_matrix)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)

synthetic_X, synthetic_y = create_synthetic_training_data()
visualize_training_data(synthetic_X, synthetic_y)

grid_data = create_dense_test_grid()
for neighbors in [1, 3, 5, 7]:
    knn_classify_and_visualize(synthetic_X, synthetic_y, grid_data, neighbors)

optimal_k, cv_accuracy = tune_knn_hyperparameter(X_tr, y_tr)
print("\nBest k value found:", optimal_k)
print("Best Cross Validation Score:", cv_accuracy)
