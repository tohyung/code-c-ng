import numpy as np

x = np.array([3, 6, 9, 10, 12, 15, 18, 20])
y = np.array([1.1, 2, 3.2, 3.4, 3.9, 5, 6.2, 6.6])

def tạo_ma_trận_X(x, degree):
    n = len(x)
    X = np.ones((n, degree + 1))
    for i in range(1, degree + 1):
        X[:, i] = x ** i
    return X

def chỉ_số_R2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


bậc_ptrinh = 10

for bậc in range(1, bậc_ptrinh + 1):
    X = tạo_ma_trận_X(x, bậc)

    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    y_pred = X @ beta

    r2 = chỉ_số_R2(y, y_pred)
    error = rmse(y, y_pred)

    equation = " + ".join([f"{round(beta[i],4)}*x^{i}" for i in range(len(beta))])

    print("="*50)
    print(f"Bậc phương trình: {bậc}")
    print(f"Phương trình: y = {equation}")
    print(f"R^2: {round(r2,6)}")
    print(f"RMSE: {round(error,6)}")
