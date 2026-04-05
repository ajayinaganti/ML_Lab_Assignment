import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Optional (if installed)
try:
    from xgboost import XGBClassifier
except:
    XGBClassifier = None


# Load Dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    dataset = dataset.drop(columns=["name"])
    features = dataset.drop("status", axis=1)
    target = dataset["status"]
    return features, target


# Preprocessing
def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features


# Split Data
def split_dataset(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)


# Model Evaluation
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)

    train_predictions = classifier.predict(X_train)
    test_predictions = classifier.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions),
        "recall": recall_score(y_test, test_predictions),
        "f1_score": f1_score(y_test, test_predictions)
    }

    return metrics


# Hyperparameter Tuning
def perform_hyperparameter_tuning(model, param_grid, X_train, y_train):
    randomized_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    randomized_search.fit(X_train, y_train)
    return randomized_search.best_estimator_


# Run Multiple Models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    all_results = {}

    # SVM
    svm_model = SVC()
    svm_param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }
    tuned_svm = perform_hyperparameter_tuning(svm_model, svm_param_grid, X_train, y_train)
    all_results["SVM"] = evaluate_classifier(tuned_svm, X_train, X_test, y_train, y_test)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_param_grid = {
        "max_depth": [3, 5, 10],
        "criterion": ["gini", "entropy"]
    }
    tuned_dt = perform_hyperparameter_tuning(dt_model, dt_param_grid, X_train, y_train)
    all_results["Decision Tree"] = evaluate_classifier(tuned_dt, X_train, X_test, y_train, y_test)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }
    tuned_rf = perform_hyperparameter_tuning(rf_model, rf_param_grid, X_train, y_train)
    all_results["Random Forest"] = evaluate_classifier(tuned_rf, X_train, X_test, y_train, y_test)

    # AdaBoost
    ada_model = AdaBoostClassifier()
    ada_param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.5, 1]
    }
    tuned_ada = perform_hyperparameter_tuning(ada_model, ada_param_grid, X_train, y_train)
    all_results["AdaBoost"] = evaluate_classifier(tuned_ada, X_train, X_test, y_train, y_test)

    # Naive Bayes
    nb_model = GaussianNB()
    all_results["Naive Bayes"] = evaluate_classifier(nb_model, X_train, X_test, y_train, y_test)

    # MLP
    mlp_model = MLPClassifier(max_iter=500)
    mlp_param_grid = {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu", "tanh"]
    }
    tuned_mlp = perform_hyperparameter_tuning(mlp_model, mlp_param_grid, X_train, y_train)
    all_results["MLP"] = evaluate_classifier(tuned_mlp, X_train, X_test, y_train, y_test)

    # XGBoost (if available)
    if XGBClassifier:
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5]
        }
        tuned_xgb = perform_hyperparameter_tuning(xgb_model, xgb_param_grid, X_train, y_train)
        all_results["XGBoost"] = evaluate_classifier(tuned_xgb, X_train, X_test, y_train, y_test)

    return all_results


# Main Function
def main():
    features, target = load_dataset("parkinsons.csv")

    features = scale_features(features)

    X_train, X_test, y_train, y_test = split_dataset(features, target)

    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Print Results
    print("\nModel Performance Comparison:\n")

    for model_name, performance in model_results.items():
        print(f"{model_name}:")
        for metric_name, metric_value in performance.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print()


if __name__ == "__main__":
    main()