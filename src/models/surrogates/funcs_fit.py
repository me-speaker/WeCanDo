'''
多项式拟合用到的一些标准评价函数
'''
import numpy as np
def stdError_func(y_test, y):
      return np.sqrt(np.mean((y_test - y) ** 2))

def R2_1_func(y_test, y):
  return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()

def R2_2_func(y_test, y):
  y_mean = np.array(y)
  y_mean[:] = y.mean()
  return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)
