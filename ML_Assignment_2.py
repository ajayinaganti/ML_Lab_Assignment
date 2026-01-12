import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx")

#selected = df.loc[0:10, ["Customer", "Candies (#)", "Mangoes (Kg)","Milk Packets (#)", "Payment (Rs)" ]]
#print(selected)

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