import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


def read_data(file_path):
    data_frame = pd.read_csv(file_path)
    return data_frame


def split_data(data_frame, target_col):
    features = data_frame.drop(columns=["name", target_col])
    labels = data_frame[target_col]

    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    return train_x, test_x, train_y, test_y


def regression_single(train_x, train_y, col):

    feature_data = train_x[[col]]

    reg_model = LinearRegression()
    reg_model.fit(feature_data, train_y)

    pred_train = reg_model.predict(feature_data)

    return reg_model, pred_train


def regression_metrics(actual, predicted):

    err_mse = mean_squared_error(actual, predicted)
    err_rmse = np.sqrt(err_mse)
    err_mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    score_r2 = r2_score(actual, predicted)

    return err_mse, err_rmse, err_mape, score_r2


def regression_multi(train_x, train_y):

    multi_model = LinearRegression()
    multi_model.fit(train_x, train_y)

    train_prediction = multi_model.predict(train_x)

    return multi_model, train_prediction


def clustering_kmeans(data_x, clusters):

    k_model = KMeans(n_clusters=clusters, random_state=42, n_init="auto")
    k_model.fit(data_x)

    return k_model.labels_, k_model.cluster_centers_


def clustering_scores(data_x, cluster_labels):

    score_sil = silhouette_score(data_x, cluster_labels)
    score_ch = calinski_harabasz_score(data_x, cluster_labels)
    score_db = davies_bouldin_score(data_x, cluster_labels)

    return score_sil, score_ch, score_db


def test_k_values(data_x, k_values):

    sil_list = []
    ch_list = []
    db_list = []

    for val in k_values:

        k_model = KMeans(n_clusters=val, random_state=42, n_init="auto")
        label_values = k_model.fit_predict(data_x)

        sil_list.append(silhouette_score(data_x, label_values))
        ch_list.append(calinski_harabasz_score(data_x, label_values))
        db_list.append(davies_bouldin_score(data_x, label_values))

    return sil_list, ch_list, db_list


def draw_plots(k_values, sil_list, ch_list, db_list):

    plt.plot(k_values, sil_list)
    plt.title("Silhouette Score vs K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.show()

    plt.plot(k_values, ch_list)
    plt.title("Calinski Harabasz Score vs K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.show()

    plt.plot(k_values, db_list)
    plt.title("Davies Bouldin Score vs K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.show()


def compute_elbow(data_x, k_values):

    inertia_vals = []

    for val in k_values:

        k_model = KMeans(n_clusters=val, random_state=42, n_init="auto")
        k_model.fit(data_x)

        inertia_vals.append(k_model.inertia_)

    return inertia_vals


def draw_elbow(k_values, inertia_vals):

    plt.plot(k_values, inertia_vals)
    plt.xlabel("K")
    plt.ylabel("Distortion")
    plt.title("Elbow Method")
    plt.show()


def run():

    dataset = read_data("parkinsons.csv")

    target_name = "PPE"

    x_train, x_test, y_train, y_test = split_data(dataset, target_name)

    col_name = x_train.columns[0]
    model_one, pred_train_one = regression_single(x_train, y_train, col_name)

    pred_test_one = model_one.predict(x_test[[col_name]])

    train_scores = regression_metrics(y_train, pred_train_one)
    test_scores = regression_metrics(y_test, pred_test_one)

    print("A1 & A2 Single Feature Regression")
    print("Train Metrics:", train_scores)
    print("Test Metrics:", test_scores)

    model_all, pred_train_all = regression_multi(x_train, y_train)

    pred_test_all = model_all.predict(x_test)

    train_scores_all = regression_metrics(y_train, pred_train_all)
    test_scores_all = regression_metrics(y_test, pred_test_all)

    print("\nA3 Multiple Feature Regression")
    print("Train Metrics:", train_scores_all)
    print("Test Metrics:", test_scores_all)

    cluster_data = dataset.drop(columns=["name", target_name])

    cluster_labels, cluster_centers = clustering_kmeans(cluster_data, 2)

    print("\nCluster Centers")
    print(cluster_centers)

    sil_val, ch_val, db_val = clustering_scores(cluster_data, cluster_labels)

    print("\nClustering Scores")
    print("Silhouette:", sil_val)
    print("CH Score:", ch_val)
    print("DB Index:", db_val)

    k_vals = range(2, 10)

    sil_vals, ch_vals, db_vals = test_k_values(cluster_data, k_vals)

    draw_plots(k_vals, sil_vals, ch_vals, db_vals)

    elbow_vals = compute_elbow(cluster_data, range(2, 20))

    draw_elbow(range(2, 20), elbow_vals)


if __name__ == "__main__":
    run()