#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from get_Errors import get_MAPE
from func_Beam_Deflection_XGB3 import get_cost
from func_Beam_Deflection_XGB3 import get_constraint1
from func_Beam_Deflection_XGB3 import get_trained_model
from math import sqrt
import pandas as pd
import xgboost as xgb
from pyswarm import pso

#Path
Path =""

#Reading data
Data = pd.read_excel(Path + "Long-Term Deflection of Reinforced Concrete Beams_New.xlsx")
Data.describe()



predictors = [x for x in Data.columns if x not in ['Y']]
target = ['Y']
numerics = [x for x in predictors if x not in ['X2']]
Data[numerics] = preprocessing.scale(Data[numerics])
Data.describe()


# Spliting training and testing dataset
spliting_ratio1 = 0.15  # Ratio of data reserved for testing
spliting_ratio2 = 0.20  # Ratio of data reserved for validating
##split = len(Data) - int(len(Data) * spliting_ratio)

# Runing 20 experiments
for i in range(20):
    DataTrain1, DataTest = train_test_split(Data, test_size=spliting_ratio1, shuffle=True)
    DataTrain2, DataValidate  = train_test_split(DataTrain1, test_size=spliting_ratio2, shuffle=True) 
    
    # Converting data into Dmatrix format (better for XGBoost)
    Xtrain1 = xgb.DMatrix(DataTrain1[predictors])
    Ytmp = DataTrain1[target].to_numpy()
    YTr = Ytmp.reshape(len(Ytmp),)
    
    Xtest = xgb.DMatrix(DataTest[predictors])
    Ytmp = DataTest[target].to_numpy()
    YTe  = Ytmp.reshape(len(Ytmp),)
    
    data_train1 = xgb.DMatrix(data=DataTrain1[predictors],label=DataTrain1[target])
    data_train2 = xgb.DMatrix(data=DataTrain2[predictors],label=DataTrain2[target])
    data_validate = xgb.DMatrix(data=DataValidate[predictors],label=DataValidate[target])
    
    
    #Setting parameters for standard XGBoost model 
    param = [ ('objective', 'reg:squarederror'),('eval_metric', 'rmse'),
                 ('nthread', 4), ('verbosity',0), ('gamma',0) ]
        
    evallist = [(data_train2, 'training'),(data_validate, 'validating')]
    evals_result = {}
    xgb_model = xgb.train(param, data_train2, num_boost_round=100, evals=evallist, evals_result=evals_result, early_stopping_rounds=25)
    
    
    # Training the standard XGBoost model and output the benchmark information
    YTr_predicted = xgb_model.predict(Xtrain1,ntree_limit=xgb_model.best_ntree_limit)
    #YTr_predicted = xgb_model.predict(data_train1)
    RMSE = sqrt(mean_squared_error(YTr,YTr_predicted))
    print('RMSE of XGB model on the training data set: {0:6.2f}'.format(RMSE))
    MAE = mean_absolute_error(YTr,YTr_predicted)
    print('MAE of XGB model on the training data set: {0:6.2f}'.format(MAE))
    MAPE = get_MAPE(YTr,YTr_predicted)
    print('MAPE of XGB model on the training data set: {0:6.2f}'.format(MAPE))
    
    R2 = r2_score(YTr,YTr_predicted)
    print('R2 of XGB model on the training data set: {0:6.2f}'.format(R2))
    
    
    print('Benchmarks using the training dataset')
    
    df1=pd.DataFrame({'Training':[RMSE, MAE, MAPE, R2]}
                     ,index=['RMSE', 'MAE', 'MAPE', 'R2'])
    
    df1.style.format("{:.4f}")                    
    
    # Testing the standard XGBoost model and output the benchmark information
    YTe_predicted = xgb_model.predict(Xtest,ntree_limit=xgb_model.best_ntree_limit)
    RMSE = sqrt(mean_squared_error(YTe,YTe_predicted))
    print('RMSE of XGB model on the testing data set: {0:6.2f}'.format(RMSE))
    MAE = mean_absolute_error(YTe,YTe_predicted)
    print('MAE of XGB model on the testing data set: {0:6.2f}'.format(MAE))
    MAPE = get_MAPE(YTe,YTe_predicted)
    print('MAPE of XGB model on the testing data set: {0:6.2f}'.format(MAPE))
    R2 = r2_score(YTe,YTe_predicted)
    print('R2 of XGB model on the testing data set: {0:6.2f}'.format(R2))
    
    print('Benchmarks using the testing dataset')
    
    df2=pd.DataFrame({'Testing':[RMSE, MAE, MAPE, R2]}
                     ,index=['RMSE', 'MAE', 'MAPE','R2'])
    
    df2.style.format("{:.4f}")       
    
    
    f=open('tuning_results.txt','a+')
    fmse1=open('rmse_default.txt','a+')
    fmae1=open('mae_default.txt','a+')
    fmape1=open('mape_default.txt','a+')
    fr2_1=open('r2_default.txt','a+')
    df=pd.merge(df1, df2, left_index=True, right_index=True)
    f.write('Results for default selection of parameters:\n')
    f.write(df.to_string())
    
    np.savetxt(fmse1, df.values[0,:], fmt='%8.4f', newline=' ')
    np.savetxt(fmae1, df.values[1,:], fmt='%8.4f', newline=' ')
    np.savetxt(fmape1, df.values[2,:], fmt='%8.4f', newline=' ')
    np.savetxt(fr2_1, df.values[3,:], fmt='%8.4f', newline=' ')
    
    f.write('\n')
    fmse1.write('\n')
    fmae1.write('\n')
    fmape1.write('\n')
    fr2_1.write('\n')
    
    fmse1.close()
    fmae1.close()
    fmape1.close()
    fr2_1.close()
   


    # Tuning XGBoost parameters using PSO
    # Set the lower and upper bounds for parameter to be tuned 
    lb = [0.01, 3, 100, 0.5]
    ub = [0.5, 10, 200, 1.0]
    
    f.write('Tuning ranges\n')    
    f.write(str(lb))
    f.write('\n\n')
    f.write(str(ub))
    f.write('\n\n')

    # Runing pso
    xopt, fopt, p, pf = pso(get_cost, lb, ub, ieqcons=[get_constraint1], kwargs={'data_train':data_train2, 'data_validate':data_validate},
                                       swarmsize=100, maxiter=200, particle_output=True)
    
    xopt
    
    f.write('Tuned parameters:\n')  
    f.write(str(xopt))
    f.write('\n\n')
   
    # Get the best (tuned) model
    best_model, evals_result =get_trained_model(xopt,data_train=data_train2,data_validate=data_validate)
    
    i=best_model.best_ntree_limit-1
    f.write('Best iteration: {0:7}\n'.format(i))
    f.write('Training rmse: {0:7.3}\n'.format(evals_result['training']['rmse'][i]))
    f.write('Validating rmse: {0:7.3}\n'.format(evals_result['validating']['rmse'][i]))
    f.write('Loss value: {0:7.3}\n'.format(evals_result['training']['rmse'][i]+evals_result['validating']['rmse'][i]))
    
    
    # Training the tuned XGBoost model and output the benchmark information
    YTr_predicted = best_model.predict(Xtrain1,ntree_limit=best_model.best_ntree_limit)
    #YTr_predicted = xgb_model.predict(data_train1)
    RMSE = sqrt(mean_squared_error(YTr,YTr_predicted))
    print('RMSE of XGB model on the training data set: {0:6.2f}'.format(RMSE))
    MAE = mean_absolute_error(YTr,YTr_predicted)
    print('MAE of XGB model on the training data set: {0:6.2f}'.format(MAE))
    MAPE = get_MAPE(YTr,YTr_predicted)
    print('MAPE of XGB model on the training data set: {0:6.2f}'.format(MAPE))
    
    R2 = r2_score(YTr,YTr_predicted)
    print('R2 of XGB model on the training data set: {0:6.2f}'.format(R2))
    
    
    print('Benchmarks using the training dataset')
    
    df1=pd.DataFrame({'Training':[RMSE, MAE, MAPE, R2]}
                     ,index=['RMSE', 'MAE', 'MAPE', 'R2'])
    
    df1.style.format("{:.4f}")                 
    
    
    # Testing the tuned XGBoost model and output the benchmark information
    YTe_predicted = best_model.predict(Xtest,ntree_limit=best_model.best_ntree_limit)
    RMSE = sqrt(mean_squared_error(YTe,YTe_predicted))
    print('RMSE of XGB model on the testing data set: {0:6.2f}'.format(RMSE))
    MAE = mean_absolute_error(YTe,YTe_predicted)
    print('MAE of XGB model on the testing data set: {0:6.2f}'.format(MAE))
    MAPE = get_MAPE(YTe,YTe_predicted)
    print('MAPE of XGB model on the testing data set: {0:6.2f}'.format(MAPE))
    R2 = r2_score(YTe,YTe_predicted)
    print('R2 of XGB model on the testing data set: {0:6.2f}'.format(R2))
    
    import pandas as pd
    print('Benchmarks using the testing dataset')
    
    df2=pd.DataFrame({'Testing':[RMSE, MAE, MAPE, R2]}
                     ,index=['RMSE', 'MAE', 'MAPE','R2'])
    
    df2.style.format("{:.4f}")       
    
    
    df=pd.merge(df1, df2, left_index=True, right_index=True)
    f.write('Results for tuned parameters:\n')
    f.write(df.to_string())
    
    fmse2=open('Deflection_of_Reinforced_Concrete_Beams/rmse_tuned.txt','a+')
    fmae2=open('Deflection_of_Reinforced_Concrete_Beams/mae_tuned.txt','a+')
    fmape2=open('Deflection_of_Reinforced_Concrete_Beams/mape_tuned.txt','a+')
    fr2_2=open('Deflection_of_Reinforced_Concrete_Beams/r2_tuned.txt','a+')
    
    np.savetxt(fmse2, df.values[0,:], fmt='%8.4f', newline=' ')
    np.savetxt(fmae2, df.values[1,:], fmt='%8.4f', newline=' ')
    np.savetxt(fmape2, df.values[2,:], fmt='%8.4f', newline=' ')
    np.savetxt(fr2_2, df.values[3,:], fmt='%8.4f', newline=' ')
    
    f.write('\n')
    f.write('#######################################################')
    f.write('\n')
    fmse2.write('\n')
    fmae2.write('\n')
    fmape2.write('\n')
    fr2_2.write('\n')
    
    
    f.close()     
    fmse2.close()
    fmae2.close()
    fmape2.close()
    fr2_2.close()

