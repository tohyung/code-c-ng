import numpy as np
import time
from sklearn.linear_model import LinearRegression

demo_data = 10000

số_lượng_biến = [10, 100, 1000, 10000]

for i in số_lượng_biến:
    print(f"\n {i} Biến:")
    
    X = np.random.rand(demo_data, i)
    true_coef = np.random.rand(i)
    y = X @ true_coef + np.random.randn(demo_data) * 0.01
    
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    print("Thời gian chạy:", round(end_time - start_time, 4), "giây")
