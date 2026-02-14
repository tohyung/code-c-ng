import matplotlib.pyplot as plt

def f(x):
    return x**2 - 2

def df(x):
    return 2*x

x = 1.5  # giá trị ban đầu
steps = [x]

for i in range(6):
    x = x - f(x)/df(x)
    steps.append(x)

print("Nghiệm gần đúng:", x)

# Vẽ minh họa
import numpy as np
x_vals = np.linspace(0, 2, 400)
y_vals = x_vals**2 - 2

plt.plot(x_vals, y_vals)
plt.axhline(0)
plt.scatter(steps, [f(s) for s in steps])
plt.title("Newton Method")
plt.show()
