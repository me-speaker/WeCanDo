#代理模型训练代码
#用多项式拟合代替CFD计算
#by沈嘉竣
'''
本篇代码功能：训练一个多项式拟合函数,函数的输入为CST参数,输出为CD和CL气动系数
本篇代码输出：一个储存所有多项式系数的文件
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import funcs_fit
#数据导入
cdldata_path = r"foils\CST-CLCD-data.xlsx"
cstdata_path = r"foils\cst_data.txt"

orin_data = pd.read_excel(cdldata_path , usecols= [1,2] , skiprows=[40 , 48]) 
cdcl_data = orin_data.values.tolist()
cdcl_data = np.array(cdcl_data)
cd_data_orin = cdcl_data[: , 0]
cl_data = cdcl_data[: , 1]
cst_data = np.loadtxt(cstdata_path)
cst_data_orin = np.delete(cst_data , (38 , 46) , axis=0)
cst_data = cst_data_orin
cd_data = cd_data_orin
#print(cst_data)
test_x = cst_data_orin[42:48 , :]
test_y = cd_data_orin[42:48]

#对数据进行多项式拟合
poly_reg = PolynomialFeatures(degree=2)
cst_poly = poly_reg.fit_transform(cst_data)
#print(cst_poly)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(cst_poly , cd_data)
predict_cd = lin_reg_2.predict(cst_poly)
strError = funcs_fit.stdError_func(predict_cd , cd_data)
R2_1 = funcs_fit.R2_1_func(predict_cd , cd_data)
R2_2 = funcs_fit.R2_2_func(predict_cd , cd_data)
score = lin_reg_2.score(cst_poly , predict_cd)

print("coefficients", lin_reg_2.coef_)
print("intercept", lin_reg_2.intercept_)
print("误差",R2_1)
save_path = r"foils\cst_cd_fitpara.txt"
np.savetxt(save_path , lin_reg_2.coef_ , fmt="%s")
with open(save_path , "a+") as file:
    np.savetxt(file , np.array([lin_reg_2.intercept_]) , fmt="%s")



