
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


from lime.lime_tabular import LimeTabularExplainer





def read_dataset(path):
    """
    Load dataset and split into features and target
    """
    df = pd.read_csv("parkinsons.csv")

    # Parkinson dataset: target column is 'status'
    features = df.drop(columns=['status', 'name'])
    target = df['status']

    return features, target


# A1: STACKING CLASSIFIER

def build_stacking_model(train_X, test_X, train_y, test_y):
    """
    Implements stacking classifier with different base models
    """

    # Base models
    estimators_list = [
        ('rf_model', RandomForestClassifier(n_estimators=100)),
        ('svm_model', SVC(probability=True))
    ]

    # Meta model
    final_model = LogisticRegression()

    # Stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators_list,
        final_estimator=final_model
    )

    # Train
    stacking_clf.fit(train_X, train_y)

    # Predict
    predictions = stacking_clf.predict(test_X)

    # Accuracy
    accuracy_val = accuracy_score(test_y, predictions)

    return stacking_clf, accuracy_val



# A2: PIPELINE IMPLEMENTATION

def build_pipeline_model(train_X, test_X, train_y, test_y):
    """
    Pipeline with scaling + classifier
    """

    model_pipeline = Pipeline([
        ('scaling_step', StandardScaler()),
        ('rf_classifier', RandomForestClassifier(n_estimators=100))
    ])

    # Train
    model_pipeline.fit(train_X, train_y)

    # Predict
    predictions = model_pipeline.predict(test_X)

    # Accuracy
    accuracy_val = accuracy_score(test_y, predictions)

    return model_pipeline, accuracy_val



# A3: LIME EXPLAINER

def generate_lime_explanation(trained_pipeline, train_X, test_X):
    """
    Explain predictions using LIME
    """

    lime_exp = LimeTabularExplainer(
        training_data=np.array(train_X),
        feature_names=train_X.columns,
        class_names=['Healthy', 'Parkinson'],
        mode='classification'
    )

    # Explain first test instance
    explanation_obj = lime_exp.explain_instance(
        test_X.iloc[0].values,
        trained_pipeline.predict_proba
    )

    return explanation_obj


# MAIN FUNCTION

def execute_program():
    # Load dataset
    file_loc = "parkinsons (1).csv"
    input_X, output_y = read_dataset(file_loc)

    # Train-test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        input_X, output_y, test_size=0.2, random_state=42
    )

    # A1: Stacking
    stacking_model_obj, stacking_accuracy = build_stacking_model(X_tr, X_te, y_tr, y_te)
    print("A1: Stacking Accuracy:", stacking_accuracy)

    # A2: Pipeline
    pipeline_obj, pipeline_accuracy = build_pipeline_model(X_tr, X_te, y_tr, y_te)
    print("A2: Pipeline Accuracy:", pipeline_accuracy)

    # A3: LIME
    lime_result = generate_lime_explanation(pipeline_obj, X_tr, X_te)
    print("A3: LIME Explanation:")
    print(lime_result.as_list())


# RUN PROGRAM

if __name__ == "__main__":
    execute_program()