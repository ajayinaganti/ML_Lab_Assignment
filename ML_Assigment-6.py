import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def read_dataset(file_path):
    data_frame = pd.read_csv(file_path)
    data_frame = data_frame.drop(columns=["name"])
    features = data_frame.drop("status", axis=1)
    target = data_frame["status"]
    return features, target


# A1: Entropy 
def calculate_entropy(target_values):
    total_samples = len(target_values)
    value_counts = Counter(target_values)
    entropy_value = 0

    for freq in value_counts.values():
        probability = freq / total_samples
        entropy_value -= probability * log2(probability)

    return entropy_value


# Equal Width Binning
def width_binning(values, num_bins=4):
    values = np.array(values)
    min_v, max_v = np.min(values), np.max(values)
    bin_size = (max_v - min_v) / num_bins

    binned_vals = np.floor((values - min_v) / bin_size)
    binned_vals[binned_vals == num_bins] = num_bins - 1

    return binned_vals.astype(int)


# Equal Frequency Binning
def frequency_binning(values, num_bins=4):
    values = np.array(values)
    quantile_points = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(values, quantile_points)

    return np.digitize(values, bin_edges[1:-1])


# General Binning
def apply_binning(values, num_bins=4, method_type="equal_width"):
    if method_type == "equal_width":
        return width_binning(values, num_bins)
    elif method_type == "equal_frequency":
        return frequency_binning(values, num_bins)
    else:
        raise ValueError("Invalid binning method")


# Apply binning to full dataset
def transform_dataset(feature_set, num_bins=4, method_type="equal_width"):
    transformed_data = feature_set.copy()

    for column in feature_set.columns:
        transformed_data[column] = apply_binning(feature_set[column], num_bins, method_type)

    return transformed_data


# A2: Gini Index 
def calculate_gini(target_values):
    total_samples = len(target_values)
    value_counts = Counter(target_values)

    gini_value = 1
    for freq in value_counts.values():
        probability = freq / total_samples
        gini_value -= probability ** 2

    return gini_value


# A3: Information Gain 
def compute_information_gain(features, target, feature_name):
    base_entropy = calculate_entropy(target)
    unique_vals = np.unique(features[feature_name])

    weighted_entropy = 0
    for val in unique_vals:
        subset_target = target[features[feature_name] == val]
        weight = len(subset_target) / len(target)
        weighted_entropy += weight * calculate_entropy(subset_target)

    return base_entropy - weighted_entropy


def select_best_feature(features, target):
    gain_dict = {}

    for column in features.columns:
        gain_dict[column] = compute_information_gain(features, target, column)

    best_col = max(gain_dict, key=gain_dict.get)
    return best_col, gain_dict


# A5: Custom Decision Tree 
class TreeNode:
    def __init__(self, split_feature=None, label=None, branches=None):
        self.split_feature = split_feature
        self.label = label
        self.branches = branches


def create_tree(features, target):
    if len(set(target)) == 1:
        return TreeNode(label=target.iloc[0])

    if features.empty:
        return TreeNode(label=Counter(target).most_common(1)[0][0])

    best_col, _ = select_best_feature(features, target)
    node = TreeNode(split_feature=best_col, branches={})

    for val in np.unique(features[best_col]):
        sub_features = features[features[best_col] == val].drop(columns=[best_col])
        sub_target = target[features[best_col] == val]

        node.branches[val] = create_tree(sub_features, sub_target)

    return node


#  A6: Visualization 
def display_tree(features, target):
    clf_model = DecisionTreeClassifier()
    clf_model.fit(features, target)

    plt.figure(figsize=(12, 6))
    plot_tree(clf_model, feature_names=features.columns, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

    return clf_model


# A7: Decision Boundary 
def plot_decision_boundary(features, target):
    clf_model = DecisionTreeClassifier()
    clf_model.fit(features, target)

    x_min, x_max = features.iloc[:, 0].min() - 1, features.iloc[:, 0].max() + 1
    y_min, y_max = features.iloc[:, 1].min() - 1, features.iloc[:, 1].max() + 1

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    predictions = clf_model.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    predictions = predictions.reshape(grid_x.shape)

    plt.contourf(grid_x, grid_y, predictions, alpha=0.4)
    plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=target)

    plt.title("Decision Boundary")
    plt.xlabel(features.columns[0])
    plt.ylabel(features.columns[1])
    plt.show()

    return clf_model


#  MAIN FUNCTION 
def run_program():
    features, target = read_dataset("parkinsons.csv")

    # Binning
    processed_features = transform_dataset(features, num_bins=4, method_type="equal_width")

    # Entropy
    ent_val = calculate_entropy(target)
    print("Entropy:", ent_val)

    # Gini
    gini_val = calculate_gini(target)
    print("Gini Index:", gini_val)

    # Best Feature
    best_feature, all_gains = select_best_feature(processed_features, target)
    print("Best Feature:", best_feature)
    print("Information Gains:", all_gains)

    # Build Tree
    decision_tree = create_tree(processed_features, target)
    print("Tree Created Successfully")

    # Visualization
    display_tree(processed_features, target)

    # Decision Boundary
    plot_decision_boundary(processed_features.iloc[:, :2], target)


# RUN 
if __name__ == "__main__":
    run_program()