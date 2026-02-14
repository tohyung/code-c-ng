import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 6, 9, 10, 12, 15, 18, 20])
x2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  
x3 = np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8])  
y  = np.array([1.1, 2.0, 3.2, 3.4, 3.9, 5.0, 6.2, 6.6])  

# tính r2 và rmse
def evaluate(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return r2, rmse

# đa thức bậc 1
X1 = np.column_stack((np.ones(len(x1)), x1, x2, x3))
beta1 = np.linalg.pinv(X1) @ y

y_pred1 = X1 @ beta1
r2_1, rmse_1 = evaluate(y, y_pred1)

# dự đoán lợi nhuận t9
x9 = np.array([1, 25, 5, 3.5])
y9_1 = x9 @ beta1

print("Đa thức bậc 1")
print(f"Phương trình:")
print(f"y = {beta1[0]:.4f} + {beta1[1]:.4f}*x1 + {beta1[2]:.4f}*x2 + {beta1[3]:.4f}*x3")
print(f"R2 = {r2_1:.4f}")
print(f"RMSE = {rmse_1:.4f}")
print(f"Dự đoán lợi nhuận tháng 9 = {y9_1:.4f}")

# đa thức bậc 2
X2 = np.column_stack((
    np.ones(len(x1)),
    x1, x2, x3,
    x1**2, x2**2, x3**2,
    x1*x2, x1*x3, x2*x3
))

beta2 = np.linalg.pinv(X2) @ y

y_pred2 = X2 @ beta2
r2_2, rmse_2 = evaluate(y, y_pred2)

# Dự đoán tháng 9
x9_poly = np.array([
    1,
    25, 5, 3.5,
    25**2, 5**2, 3.5**2,
    25*5, 25*3.5, 5*3.5
])
y9_2 = x9_poly @ beta2

print("\nĐa thức bậc 2")
print("Phương trình:")
print("y =", " + ".join([f"{beta2[i]:.4f}*X{i}" for i in range(len(beta2))]))
print(f"R2 = {r2_2:.4f}")
print(f"RMSE = {rmse_2:.4f}")
print(f"Dự đoán lợi nhuận tháng 9 = {y9_2:.4f}")


# biểu đồ
x3_mean = np.mean(x3)

x1_range = np.linspace(min(x1), max(x1), 100)
x2_range = np.interp(x1_range, x1, x2)

# bậc 1
y_plot1 = (
    beta1[0]
    + beta1[1]*x1_range
    + beta1[2]*x2_range
    + beta1[3]*x3_mean
)

plt.figure()
plt.scatter(x1, y)
plt.plot(x1_range, y_plot1)
plt.title("Đa thức bậc 1 (3 biến, vẽ theo 2 biến)")
plt.xlabel("Doanh thu")
plt.ylabel("Lợi nhuận")
plt.show()


# bậc 2
y_plot2 = (
    beta2[0]
    + beta2[1]*x1_range
    + beta2[2]*x2_range
    + beta2[3]*x3_mean
    + beta2[4]*x1_range**2
    + beta2[5]*x2_range**2
    + beta2[6]*x3_mean**2
    + beta2[7]*x1_range*x2_range
    + beta2[8]*x1_range*x3_mean
    + beta2[9]*x2_range*x3_mean
)

plt.figure()
plt.scatter(x1, y)
plt.plot(x1_range, y_plot2)
plt.title("Đa thức bậc 2 (3 biến, vẽ theo 2 biến)")
plt.xlabel("Doanh thu")
plt.ylabel("Lợi nhuận")
plt.show()
