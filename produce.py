import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()

# 給定隨機種子，使每次執行結果保持一致
np.random.seed(1)


def getdata(n):
    x = np.arange(-5, 5.1, 10/(n-1))
    # 給定一個固定的參數，再加上隨機變動值作為雜訊，其變動值介於 +-10 之間
    y = 3*x + 2 + (np.random.rand(len(x))-0.5)*20
    return x, y

def plot_error(x, y):
    a = np.arange(-10, 16, 1)
    b = np.arange(-10, 16, 1)
    mesh = np.meshgrid(a, b)

    sqr_err = 0
    for xs, ys in zip(x, y):
        sqr_err += ((mesh[0]*xs + mesh[1]) - ys) ** 2
    loss = sqr_err/len(x)
    
    plt.contour(mesh[0], mesh[1], loss, 20, cmap=plt.cm.jet)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.axis('scaled')
    plt.title('function loss')

class my_BGD:    
    def __init__(self, a, b, x, y, alpha):
        self.a = a
        self.b = b
        self.x = x
        self.y = y
        self.alpha = alpha
        
        self.a_old = a
        self.b_old = b
        
        self.loss = self.mse();
    
    # Loss function
    def mse(self):
        sqr_err = ((self.a*self.x + self.b) - self.y) ** 2
        return np.mean(sqr_err)
    
    def gradient(self):
        grad_a = 2 * np.mean((self.a*self.x + self.b - self.y) * (self.x))
        grad_b = 2 * np.mean((self.a*self.x + self.b - self.y) * (1))
        return grad_a, grad_b

    def update(self):
        grad_a, grad_b = self.gradient()
        self.a_old = self.a
        self.b_old = self.b
        self.a = self.a - self.alpha * grad_a
        self.b = self.b - self.alpha * grad_b
        self.loss = self.mse();

x, y = getdata(51)
data = pd.DataFrame({"X":x,"Y":y})
data.to_excel('sample_data.xlsx', sheet_name='sheet1', index=False)
xl,yl=getdata(51)
fig2 = plt.figure(num=2)
plt.plot(xl, yl, 'bo')
plt.grid('true')
plt.ylim(-30, 30)
plt.xlabel('x')
plt.ylabel('f(x)')
#plt.show()
# 繪製誤差空間底圖
fig1 = plt.figure(num=1)
plot_error(x, y)
#plt.show()


alpha = 0.1

a = -9; b = -9

# 初始化
mlclass = my_BGD(a, b, x, y, alpha)

plt.plot(a, b, 'ro-')
plt.title('Initial, loss='+'{:.2f}'.format(mlclass.loss)+'\na='+
          '{:.2f}'.format(a)+', b='+'{:.2f}'.format(b))

for i in range(1, 101):
    mlclass.update()
    print('iter='+str(i)+', loss='+'{:.2f}'.format(mlclass.loss))
    fig1 = plt.figure(num=1)
    plt.plot((mlclass.a_old, mlclass.a), (mlclass.b_old, mlclass.b), 'ro-')
    plt.title('iter='+str(i)+', loss='+'{:.2f}'.format(mlclass.loss)+'\na='+'{:.2f}'.format(mlclass.a)+', b='+'{:.2f}'.format(mlclass.b))
    #plt.show()
    fig1.savefig("test/b"+str(i)+'.png')
    fig2 = plt.figure(num=i+3)
    xf=np.linspace(-5, 5, 100) 
    yf=mlclass.a_old*xf+mlclass.b_old
    plt.plot(xl, yl, 'bo')
    #print(xf,yf)
    plt.plot(xf, yf)
    plt.grid('true')
    plt.ylim(-30, 30)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    fig2.savefig("test/c"+str(i)+'.png')
    print(mlclass.a_old,mlclass.b_old)
    #plt.show()