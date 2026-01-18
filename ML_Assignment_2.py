import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("Lab Session Data.xlsx")
X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()
print(X)

print(np.linalg.matrix_rank(X))
X_inv = np.linalg.pinv(X)

Y = df[["Payment (Rs)"]].to_numpy()

C = X_inv @ Y
print("Price of each item is:\n",C)

for i in range(10):
    if Y[i][0]>200:
        print(f"Customer-{i+1} is rich")
    else:
        print(f"Customer-{i+1} is poor")

df = pd.read_excel("Lab Session Data.xlsx", sheet_name=1)
print(df)

price = df[["Price"]].to_numpy()

print(np.mean(price))
print(np.var(price))

def manual_mean(data):
    total = 0
    for x in data:
        total += x
    return total / len(data)

def manual_variance(data, mean):
    total = 0
    for x in data:
        total += (x - mean) ** 2
    return total / len(data)

def execution_time(func, data, runs=10):
    times = []
    for _ in range(runs):
        start = time.time()
        func(data)
        end = time.time()
        times.append(end - start)
    return sum(times) / runs

def jaccard_coefficient(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    denom = f01 + f10 + f11
    return 0 if denom == 0 else f11 / denom

def smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    return (f11 + f00) / (f00 + f01 + f10 + f11)

def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():

    # A1 
    df = pd.read_excel("Lab Session Data.xlsx")

    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()
    Y = df[["Payment (Rs)"]].to_numpy()

    print("\n--- A1 RESULTS ---")
    print("Feature Matrix X:\n", X)
    print("Rank of X:", np.linalg.matrix_rank(X))

    X_pinv = np.linalg.pinv(X)
    cost = X_pinv @ Y
    print("Cost of items (Candies, Mangoes, Milk):\n", cost)

    # A2
    print("\n--- A2 RESULTS ---")
    for i in range(len(Y)):
        if Y[i][0] > 200:
            print(f"Customer-{i+1}: RICH")
        else:
            print(f"Customer-{i+1}: POOR")

    # A3
    stock_df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    price = stock_df["Price"].to_numpy()

    pkg_mean = np.mean(price)
    pkg_var = np.var(price)

    man_mean = manual_mean(price)
    man_var = manual_variance(price, man_mean)

    mean_time = execution_time(manual_mean, price)
    var_time = execution_time(lambda x: manual_variance(x, man_mean), price)

    wednesday_data = stock_df[stock_df["Day"] == "Wednesday"]["Price"]
    april_data = stock_df[stock_df["Month"] == "Apr"]["Price"]

    loss_prob = len(list(filter(lambda x: x < 0, stock_df["Chg%"]))) / len(stock_df)

    profit_wed = stock_df[stock_df["Day"] == "Wednesday"]
    if len(profit_wed) == 0:
        profit_given_wed = 0
    else:
        profit_given_wed = len(profit_wed[profit_wed["Chg%"] > 0]) / len(profit_wed)
    print("\n--- A3 RESULTS ---")
    print("Package Mean:", pkg_mean)
    print("Package Variance:", pkg_var)
    print("Manual Mean:", man_mean)
    print("Manual Variance:", man_var)
    print("Mean Execution Time:", mean_time)
    print("Variance Execution Time:", var_time)
    print("Wednesday Mean:", wednesday_data.mean())
    print("April Mean:", april_data.mean())
    print("Probability of Loss:", loss_prob)
    print("Conditional Profit Probability (Wednesday):", profit_given_wed)
    plt.scatter(stock_df["Day"], stock_df["Chg%"])
    plt.title("Chg% vs Day")
    plt.show()

    # A4
    thyroid_df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    print("\n--- A4 RESULTS ---")
    print("Missing Values:\n", thyroid_df.isnull().sum())
    print("Mean:\n", thyroid_df.mean(numeric_only=True))
    print("Variance:\n", thyroid_df.var(numeric_only=True))

    # A5
    binary_df = thyroid_df.select_dtypes(include=["int64", "bool"]).fillna(0)

    v1 = binary_df.iloc[0].to_numpy()
    v2 = binary_df.iloc[1].to_numpy()

    print("\n--- A5 RESULTS ---")
    print("Jaccard Coefficient:", jaccard_coefficient(v1, v2))
    print("Simple Matching Coefficient:", smc(v1, v2))

    # A6
    full_df = thyroid_df.select_dtypes(include=["int64", "float64"]).fillna(0)
    fv1 = full_df.iloc[0].to_numpy()
    fv2 = full_df.iloc[1].to_numpy()
    print("\n--- A6 RESULT ---")
    print("Cosine Similarity:", cosine_similarity(fv1, fv2))

    # A7 
    subset = binary_df.iloc[:20].to_numpy()
    n = len(subset)

    jc_mat = np.zeros((n, n))
    smc_mat = np.zeros((n, n))
    cos_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            jc_mat[i][j] = jaccard_coefficient(subset[i], subset[j])
            smc_mat[i][j] = smc(subset[i], subset[j])
            cos_mat[i][j] = cosine_similarity(subset[i], subset[j])

    sns.heatmap(jc_mat)
    plt.title("Jaccard Coefficient Heatmap")
    plt.show()

    sns.heatmap(smc_mat)
    plt.title("SMC Heatmap")
    plt.show()

    sns.heatmap(cos_mat)
    plt.title("Cosine Similarity Heatmap")
    plt.show()

main()
