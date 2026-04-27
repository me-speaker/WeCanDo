#共轭梯度算法函数
#by沈嘉竣
'''
此篇代码为写成函数形式可api调用的CG算法
'''
from sympy import diff , symbols , solve , Array
import numpy as np
def CG_FR_free(X_init , X ,epsilon , f , iter_num):
    '''
    无约束的纯粹CG算法
    X_init:优化开始时给的初始X坐标
    X:包含所有方程自变量的矩阵
    f:方程
    iter_num:迭代次数，建议为自变量数目的倍数
    '''
    N = len(X_init)
    X_past = np.empty([N , 1]) #储存k-1步的X数值
    X_cur = np.empty([N , 1]) #储存k步的X数值
    X_new = np.empty([N , 1]) #储存k+1步的X数值
    #计算并储存各个变量对应的梯度函数 
    grad = [-diff(f , X[i]) for i in range(N)] 
    #print(grad)
    #开始进行N轮迭代
    for k in range(iter_num): #k迭代次数
        if k == 0:
            X_cur = [X_init[i] for i in range(N)] #对第一次迭代的X进行赋值
            X_input_grad_cur = [(X[n] , X_cur[n]) for n in range(N)]
        #计算步长系数和梯度方向
        else:
            X_input_grad_cur = [(X[n] , X_cur[n]) for n in range(N)] 
        grad_cur = np.array([grad[i].subs(X_input_grad_cur) for i in range(N)]) #k步的梯度的相反数
        grad_cur = grad_cur.reshape([N , 1]).astype('float')
        norm_cur = np.linalg.norm(grad_cur , ord=2 , axis=0) #第k步梯度范数
        #根据流程，k==0时梯度方向直接为梯度相反数
        #print(X_cur)
        if not k%N : #采用n步重启策略
            Dir_cur = grad_cur
        #k >= 0时根据梯度方向公式计算
        else:
            X_input_grad_past = [(X[n] , X_past[n]) for n in range(N)]
            grad_past = np.array([grad[i].subs(X_input_grad_past) for i in range(N)])
            grad_past = grad_past.reshape([N , 1]).astype('float')
            norm_past = np.linalg.norm(grad_past , ord=2 , axis=0)
            β = norm_cur**2/norm_past**2
            Dir_cur = grad_cur + β * Dir_past
        #判断是否收敛
        if norm_cur <= epsilon: #若收敛则退出
            print(norm_cur)
            print("算法达到最优")
            break
        else: #若未收敛继续迭代
            #计算步长
            λ = symbols("λ")    #步长
            X_new_sym = [X_cur[i] + λ * Dir_cur[i] for i in range(N)] #k+1步的X符号表达式
            X_new_sym = Array(X_new_sym)
            X_input_get_λ = [(X[n] , X_new_sym[n][0]) for n in range(N)] #X_new_sym为列表，需要两个索引将表达式提取出来
            f_new = f.subs(X_input_get_λ) #将x替换为λ使得f为关于λ的表达式
            grad_lamda = diff(f_new , λ)
            λ_value = solve(grad_lamda , λ)[0] #求解最优梯度值
            #基于步长更新迭代X和Dir
            X_new = X_new_sym.subs(λ , λ_value) 
            X_past = X_cur
            X_cur = X_new[: , 0] #不知道为什么，X_cur是一个3个维度的列表，这里降下维           
            Dir_past = Dir_cur
            print(norm_cur)
            
    return X_cur

def CG_FR_constrain(X_init , X ,epsilon , f , iter_num):
    '''
    针对翼型优化问题的有约束优化
    '''
    N = len(X_init)
    X_past = np.empty([N , 1]) #储存k-1步的X数值
    X_cur = np.empty([N , 1]) #储存k步的X数值
    X_new = np.empty([N , 1]) #储存k+1步的X数值
    #计算并储存各个变量对应的梯度函数 
    grad = [-diff(f , X[i]) for i in range(N)] 
    print(grad)
    #print(grad)
    #开始进行N轮迭代
    for k in range(iter_num): #k迭代次数
        if k == 0:
            X_cur = [X_init[i] for i in range(N)] #对第一次迭代的X进行赋值
            X_input_grad_cur = [(X[n] , X_cur[n]) for n in range(N)]
        #计算步长系数和梯度方向
        else:
            X_input_grad_cur = [(X[n] , X_cur[n]) for n in range(N)] 
        grad_cur = np.array([grad[i].subs(X_input_grad_cur) for i in range(N)]) #k步的梯度的相反数
        grad_cur = grad_cur.reshape([N , 1]).astype('float')
        norm_cur = np.linalg.norm(grad_cur , ord=2 , axis=0) #第k步梯度范数
        #根据流程，k==0时梯度方向直接为梯度相反数
        #print(X_cur)
        if not k%N : #采用n步重启策略
            Dir_cur = grad_cur
        #k >= 0时根据梯度方向公式计算
        else:
            X_input_grad_past = [(X[n] , X_past[n]) for n in range(N)]
            grad_past = np.array([grad[i].subs(X_input_grad_past) for i in range(N)])
            grad_past = grad_past.reshape([N , 1]).astype('float')
            norm_past = np.linalg.norm(grad_past , ord=2 , axis=0)
            β = norm_cur**2/norm_past**2
            Dir_cur = grad_cur + β * Dir_past
        #判断是否收敛
        if norm_cur <= epsilon: #若收敛则退出
            print("算法达到最优")
            break
        else: #若未收敛继续迭代
            #计算步长
            λ = symbols("λ")    #步长
            X_new_sym = [X_cur[i] + λ * Dir_cur[i] for i in range(N)] #k+1步的X符号表达式
            X_new_sym = Array(X_new_sym)
            X_input_get_λ = [(X[n] , X_new_sym[n][0]) for n in range(N)] #X_new_sym为列表，需要两个索引将表达式提取出来
            f_new = f.subs(X_input_get_λ) #将x替换为λ使得f为关于λ的表达式
            grad_lamda = diff(f_new , λ)
            λ_value = solve(grad_lamda , λ)[0] #求解最优梯度值
            #基于步长更新迭代X和Dir
            X_new = X_new_sym.subs(λ , λ_value)
            #print(X_new)                                     
            X_past = X_cur 
            X_cur = X_new[: , 0] #不知道为什么，X_cur是一个3个维度的列表，这里降下维
            X_cur = np.array(X_cur)
            for i in range(N):
                if X_cur[i] < 0.8*X_init[i]:
                    X_cur[i] = 0.8*X_init[i]
                elif  X_cur[i] > 1.2*X_init[i]:
                    X_cur[i] = 1.2*X_init[i]             
            Dir_past = Dir_cur
            print(norm_cur)
    return X_cur