def f(x):
    return x**2 - 2

def df(x):
    return 2*x

x = 1.5
steps = [x]

for i in range(6):
    x = x - f(x)/df(x)
    steps.append(x)

print("Nghiệm xấp xỉ:", x)
