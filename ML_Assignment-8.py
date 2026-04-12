
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# A1 


def summation(x, w, b):
    return np.dot(x, w) + b

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return x if x > 0 else 0.01*x

def error(t, o):
    return t - o

# Perceptron Training


def train_perceptron(X, y, lr, activation, epochs=1000):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features)
    b = np.random.randn()

    errors = []

    for ep in range(epochs):
        total_error = 0

        for i in range(n_samples):
            net = summation(X[i], w, b)
            out = activation(net)
            e = error(y[i], out)

            w += lr * e * X[i]
            b += lr * e
            total_error += e**2

        mse = total_error / n_samples
        errors.append(mse)

        if mse <= 0.002:
            break

    return w, b, errors


def predict(X, w, b, activation):
    return np.array([activation(summation(x, w, b)) for x in X])


# A2

def AND_gate():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    return X, y

# A4

def learning_rate_experiment(X, y):
    rates = np.arange(0.1, 1.1, 0.1)
    iterations = []

    for lr in rates:
        _, _, errors = train_perceptron(X, y, lr, step)
        iterations.append(len(errors))

    return rates, iterations

# A5

def XOR_gate():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    return X, y


# A6


def customer_data():
    data = np.array([
        [20,6,2,386,1],
        [16,3,6,289,1],
        [27,6,2,393,1],
        [19,1,2,110,0],
        [24,4,2,280,1],
        [22,1,5,167,0],
        [15,4,2,271,1],
        [18,4,2,274,1],
        [21,1,4,148,0],
        [16,2,4,198,0]
    ])
    X = data[:,:-1]
    y = data[:,-1]

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

# A7

def pseudo_inverse_solution(X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.linalg.pinv(X_bias).dot(y)
    return w


# A8


def backprop_AND(lr=0.05, epochs=1000):
    X, y = AND_gate()

    # Initialize weights
    w1 = np.random.randn(2,2)
    w2 = np.random.randn(2,1)

    errors = []

    for ep in range(epochs):
        total_error = 0

        for i in range(len(X)):
            x = X[i].reshape(1,-1)
            target = y[i]

            # Forward
            h = sigmoid(np.dot(x, w1))
            o = sigmoid(np.dot(h, w2))

            # Error
            e = target - o
            total_error += e**2

            # Backward
            d_o = e * o * (1 - o)
            d_h = h * (1 - h) * np.dot(d_o, w2.T)

            # Update
            w2 += lr * h.T.dot(d_o)
            w1 += lr * x.T.dot(d_h)

        mse = total_error / len(X)
        errors.append(mse)

        if mse <= 0.002:
            break

    return errors


# A10

def two_output_AND():
    X, y = AND_gate()
    y_new = np.array([[1,0] if i==0 else [0,1] for i in y])
    return X, y_new


# A12: PARKINSON DATA


def load_parkinsons(path):
    df = pd.read_csv(path)

    if 'name' in df.columns:
        df = df.drop(columns=['name'])

    y = df['status'].values
    X = df.drop(columns=['status']).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y



# Main Function


if __name__ == "__main__":

    # A2 AND Gate
    X_and, y_and = AND_gate()
    w, b, err = train_perceptron(X_and, y_and, 0.05, step)
    print("A2 AND Gate Epochs:", len(err))

    # A3 Activation comparison
    for act in [bipolar_step, sigmoid, relu]:
        _, _, e = train_perceptron(X_and, y_and, 0.05, act)
        print("Activation:", act.__name__, "Epochs:", len(e))

    # A4 Learning rate
    rates, iters = learning_rate_experiment(X_and, y_and)
    plt.plot(rates, iters)
    plt.xlabel("Learning Rate")
    plt.ylabel("Iterations")
    plt.title("Learning Rate vs Iterations")
    plt.show()

    # A5 XOR
    X_xor, y_xor = XOR_gate()
    _, _, e_xor = train_perceptron(X_xor, y_xor, 0.05, step)
    print("A5 XOR Epochs:", len(e_xor))

    # A6 Customer data
    X_c, y_c = customer_data()
    w, b, _ = train_perceptron(X_c, y_c, 0.05, sigmoid)
    preds = predict(X_c, w, b, sigmoid)
    print("A6 Accuracy:", np.mean((preds>0.5)==y_c))

    # A7 Pseudo inverse
    w_pi = pseudo_inverse_solution(X_c, y_c)
    print("A7 Pseudo-inverse weights:", w_pi)

    # A8 Backprop
    err_bp = backprop_AND()
    print("A8 Backprop epochs:", len(err_bp))

    # A10 Two output
    X2, y2 = two_output_AND()
    print("A10 Sample Output:", y2[:3])

    # A11 MLP AND/XOR
    mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    mlp.fit(X_and, y_and)
    print("A11 AND Accuracy:", mlp.score(X_and, y_and))

    mlp.fit(X_xor, y_xor)
    print("A11 XOR Accuracy:", mlp.score(X_xor, y_xor))

    # A12 Parkinson Dataset
    Xp, yp = load_parkinsons("parkinsons.csv")
    mlp.fit(Xp, yp)
    print("A12 Parkinson Accuracy:", mlp.score(Xp, yp))