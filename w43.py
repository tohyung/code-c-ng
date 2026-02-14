import numpy as np
import matplotlib.pyplot as plt


x1 = np.array([3, 6, 9, 10, 12, 15, 18, 20])          
x2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
x3 = np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8])
y  = np.array([1.1, 2.0, 3.2, 3.4, 3.9, 5.0, 6.2, 6.6])


X = np.column_stack((np.ones(len(x1)), x1, x2, x3))

beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("HỆ SỐ HỒI QUY 3 BIẾN")
print(f"a0 = {beta[0]:.4f}")
print(f"a1 = {beta[1]:.4f}")
print(f"a2 = {beta[2]:.4f}")
print(f"a3 = {beta[3]:.4f}")

# rmse
y_pred = X @ beta
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"\nRMSE = {rmse:.4f}")


x_new = np.array([1, 25, 5, 3.5])  # [1, x1, x2, x3]
y_new = x_new @ beta
print(f"\nDự báo lợi nhuận tháng 9 = {y_new:.4f}")


X_12 = np.column_stack((np.ones(len(x1)), x1, x2))
beta_12 = np.linalg.inv(X_12.T @ X_12) @ X_12.T @ y
y_pred_12 = X_12 @ beta_12

plt.figure()
plt.scatter(x1, y)
plt.plot(x1, y_pred_12)
plt.xlabel("Doanh thu (x1)")
plt.ylabel("Lợi nhuận (y)")
plt.title("Sơ đồ Doanh thu & Marketing")
plt.show()


X_13 = np.column_stack((np.ones(len(x1)), x1, x3))
beta_13 = np.linalg.inv(X_13.T @ X_13) @ X_13.T @ y
y_pred_13 = X_13 @ beta_13

plt.figure()
plt.scatter(x1, y)
plt.plot(x1, y_pred_13)
plt.xlabel("Doanh thu (x1)")
plt.ylabel("Lợi nhuận (y)")
plt.title("Sơ đồ Doanh thu & Chi lương")
plt.show()
