import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# LIME & SHAP
from lime.lime_tabular import LimeTabularExplainer
import shap


# Load Data

def load_dataset(path):
    df = pd.read_csv('parkinsons.csv')

    features = df.drop(columns=['status', 'name'])
    target = df['status']

    return features, target


# A1: Correlation Heatmap

def generate_heatmap(features):
   
    corr_data = features.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")

    return corr_data


# A2: PCA (99% variance)

def pca_model_99(train_X, test_X, train_y, test_y):
    

    scaler_obj = StandardScaler()
    train_scaled = scaler_obj.fit_transform(train_X)
    test_scaled = scaler_obj.transform(test_X)

    pca_obj = PCA(n_components=0.99)
    train_pca = pca_obj.fit_transform(train_scaled)
    test_pca = pca_obj.transform(test_scaled)

    rf_model = RandomForestClassifier()
    rf_model.fit(train_pca, train_y)

    predictions = rf_model.predict(test_pca)
    accuracy = accuracy_score(test_y, predictions)

    return accuracy, train_pca.shape[1]


# A3: PCA (95% variance)

def pca_model_95(train_X, test_X, train_y, test_y):
   

    scaler_obj = StandardScaler()
    train_scaled = scaler_obj.fit_transform(train_X)
    test_scaled = scaler_obj.transform(test_X)

    pca_obj = PCA(n_components=0.95)
    train_pca = pca_obj.fit_transform(train_scaled)
    test_pca = pca_obj.transform(test_scaled)

    rf_model = RandomForestClassifier()
    rf_model.fit(train_pca, train_y)

    predictions = rf_model.predict(test_pca)
    accuracy = accuracy_score(test_y, predictions)

    return accuracy, train_pca.shape[1]


# A4: Sequential Feature Selection

def feature_selection_sfs(train_X, test_X, train_y, test_y):
    

    rf_model = RandomForestClassifier()

    sfs_selector = SequentialFeatureSelector(
        rf_model,
        n_features_to_select=10,
        direction='forward'
    )

    sfs_selector.fit(train_X, train_y)

    train_selected = sfs_selector.transform(train_X)
    test_selected = sfs_selector.transform(test_X)

    rf_model.fit(train_selected, train_y)

    predictions = rf_model.predict(test_selected)
    accuracy = accuracy_score(test_y, predictions)

    return accuracy, train_selected.shape[1]


# A5: LIME EXPLANATION

def get_lime_explanation(trained_model, train_X, test_X):
    lime_exp = LimeTabularExplainer(
        training_data=np.array(train_X),
        feature_names=train_X.columns,
        class_names=['Healthy', 'Parkinson'],
        mode='classification'
    )

    explanation = lime_exp.explain_instance(
        test_X.iloc[0].values,
        trained_model.predict_proba
    )

    return explanation


# A5: SHAP EXPLANATION

def get_shap_values(trained_model, train_X):
    shap_exp = shap.Explainer(trained_model, train_X)
    shap_vals = shap_exp(train_X)

    return shap_vals


def main():
    dataset_path = "parkinsons (1).csv"

    input_features, output_target = load_dataset(dataset_path)

    X_tr, X_te, y_tr, y_te = train_test_split(
        input_features, output_target, test_size=0.2, random_state=42
    )

    # A1
    generate_heatmap(input_features)
    print("A1: Correlation heatmap displayed")

    # A2
    acc99, comp99 = pca_model_99(X_tr, X_te, y_tr, y_te)
    print("A2: PCA 99% -> Accuracy:", acc99, "Features:", comp99)

    # A3
    acc95, comp95 = pca_model_95(X_tr, X_te, y_tr, y_te)
    print("A3: PCA 95% -> Accuracy:", acc95, "Features:", comp95)

    # A4
    acc_sfs, feat_sfs = feature_selection_sfs(
        X_tr, X_te, y_tr, y_te
    )
    print("A4: SFS -> Accuracy:", acc_sfs, "Features:", feat_sfs)

    # Base model for explanation
    base_model = RandomForestClassifier()
    base_model.fit(X_tr, y_tr)

    # A5: LIME
    lime_result = get_lime_explanation(base_model, X_tr, X_te)
    print("A5: LIME Explanation:", lime_result.as_list())

    # A5: SHAP
    shap_result = get_shap_values(base_model, X_tr)
    print("A5: SHAP values computed")

    # SHAP plot
    shap.summary_plot(shap_result, X_tr)


if __name__ == "__main__":
    main()