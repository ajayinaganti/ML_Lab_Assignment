import numpy as np
import math

def manual_dot(A,B):
    C = []
    for i in range(len(A)):
        C.append(A[i]*B[i])
    return sum(C)

def euclidean_norm(A):
    X = []
    for x in A:
        X.append(x*x)
    return math.sqrt(sum(X))

def manual_mean(A):
    sum = 0
    for x in A:
        sum += x
    return sum / len(A)

def manual_variance(A):
    mu = manual_mean(A)
    s = 0
    for x in A:
        s += (x - mu) * (x - mu)
    return s / len(A)

def manual_std_deviation(X):
    return math.sqrt(manual_variance(X))

def mean_vector(matrix):
    no_of_cols = len(matrix[0])
    mean_vec = []

    for j in range(no_of_cols):
        col = []
        for i in range(len(matrix)):
            col.append(matrix[i][j])
        mean_vec.append(manual_mean(col))

    return mean_vec

