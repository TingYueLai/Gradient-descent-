# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure() #定義一個圖像窗口

# 學習率
alpha = 0.1

# 初始位置
w_init = 5

x = np.arange(-6, 6.1, 0.1)
y = x**2
plt.plot(x, y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x)=x^2')
plt.grid(True)

w_old = 0; w_new = 0
for i in range(1, 15):
    if i == 1:
        w_old = w_init
    else:
        w_old = w_new
    
    # (2*w_old) 為二次函數的的微分式
    w_new = w_old - alpha * (2*w_old)
    
    plt.plot((w_old, w_new), (w_old**2, w_new**2), 'ro-')
    plt.title('f(x)=x^2, error=' + '{:.2f}'.format(abs(w_new**2)))
plt.show()

fig.savefig('3.png')