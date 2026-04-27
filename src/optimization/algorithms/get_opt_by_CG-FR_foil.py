#共轭梯度优化算法
#by：沈嘉竣

'''
此篇代码读取多项式拟合得到的各项系数并重新生成一个符号多项式,再调用CG算法进行最优化求解
'''
import numpy as np
from sympy import Array , symbols , Matrix
from funcs_CG import *
#读取数据
N = 14 #输入维度
X = symbols('x1:' + str(N+1))
para_path = r"foils\cst_cd_fitpara.txt"
paras = np.loadtxt(para_path)
#将数据储存为系数矩阵方便多项式构建
c = paras[-1] + paras[0]
b = paras[1:N+1] #(N,)
b = np.reshape(b , [1 , 14])
a = np.zeros([14,14])
a_paras = paras[N+1:-1]
    #创建平方项的上三角矩阵
for i in range(N):
    for j in range(N):
        row = i+1
        col = j+1
        if row>col:
            continue
        if row<=col:
            index = (row-1)*N+col-row*(row-1)/2-1
            a[i,j] = a_paras[int(index)]
#创建多项式
X_matrix = Matrix(X)  #(N,1)
#print(X_matrix)
cd_fit = X_matrix.T*a*X_matrix + b*X_matrix + c #注意这里的cd_fit是numpy数组类型
'''
# #验证多项式是否正确
# cd_fit = Array(cd_fit) #将cd_fit转化为sympy类型
# cstdata_path = r"E:\CST\foils\cst_data.txt"
# cst_data = np.loadtxt(cstdata_path)
# test_cst = cst_data[0 , :]
# cst_input = [(X[i] , test_cst[i]) for i in range(N)]
# print(cst_input)
# print(cd_fit.subs(cst_input))
'''
#开始进行基于共轭梯度算法的优化
#rae2822_cst为基准rae2822的cst参数
rae2822_cst = [0.1265762733510324, 0.13785737028466377, 0.16165119019849966, 0.1641827785061083, 0.2204702495180052, 0.17190715612212848, 0.21320535088922354, -0.13198685777621713, -0.1246755876719183, -0.18730881956898035, -0.1511541353743263, -0.17075099564609975, 0.01841997735745539, -0.012206506878329451]
#引入惩罚函数,上面被注释掉的的是之前试过的不归一化的惩罚函数,因为各个cst参数的量级不同,这个方法效果不好,遂采用下面的归一化惩罚函数
# cst_mat = Matrix(rae2822_cst)
# cd_fit = cd_fit + (X_matrix - cst_mat).T*(X_matrix - cst_mat) #不归一化的惩罚函数
punish_weigt = 0.07 #惩罚函数系数,更改大小可以调整扰动程度
for i in range(N):
    cd_fit = cd_fit + punish_weigt*((X[i] - rae2822_cst[i])/rae2822_cst[i])**2 
#print(cd_fit)
cd_fit = Array(cd_fit)
#cst_data = [0.10427215293677568, 0.11657505544478357, 0.17383410454431866, 0.19337659491967377, 0.24968958600189176, 0.1683929901766473, 0.21402135603562697, -0.10804381118525809, -0.10255035578719411, -0.1731346830840675, -0.16158507789771048, -0.2007808695024924, 0.0214874912082598, -0.010002196641858498]
epsilon = 0.001 #精度
cst_opt = CG_FR_free(rae2822_cst , X , epsilon , cd_fit[0][0] , 2*N)
print(cst_opt)
cst_opt = np.array(cst_opt)
cst_base = np.array(rae2822_cst)
cst_max = abs(cst_opt - cst_base)*(1/cst_base)
quality = max(abs(cst_opt - cst_base)*(1/cst_base))
position = np.argmax(cst_max)
print('最大扰动为{:.2%}'.format(quality) , '位置为{}'.format(position))


    
        

    
    