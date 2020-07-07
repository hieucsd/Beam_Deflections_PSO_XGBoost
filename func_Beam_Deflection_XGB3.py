import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from get_Errors import get_MAPE
import pandas as pd
import numpy as np

# Dealing with integer constraint (maxdepth)
def get_constraint1(x,**kwargs):
    max_depth = np.int32(x[1])
    return 1e-2-(max_depth-x[1])**2
       
# Setting the parameters
def set_param(x):
    eta = x[0]
    max_depth = np.int32(x[1])
    
    n = len(x)
    if (n>2):  
        lambda_ = x[2]
    else:
        lambda_ = 1.0    

    if (n>3):  
        subsample = x[3]
    else:
        subsample = 1.0
        
    if (n>4):  
        gamma = x[4]
    else:
        gamma = 0.0
   
    if (n>5):  
        alpha = x[5]
    else:
        alpha = 0.0    
    
    param = [('eta', eta),
             ('max_depth', max_depth), 
             ('gamma',gamma), 
             ('subsample', subsample),
             ('alpha',50), 
             ('objective', 'reg:squarederror'),('eval_metric', 'rmse'),
             ('nthread', 4), ('verbosity',0) ]
    return param

# Return the cost function    
def get_cost(x,**kwargs):
    
    param = set_param(x)
    
    data_train = kwargs['data_train']
    data_validate = kwargs['data_validate'] 
    

    evallist = [(data_train, 'training'),(data_validate, 'validating')]
    evals_result = {}
    
    xgb_model = xgb.train(param, data_train, num_boost_round=500, evals=evallist, evals_result=evals_result, verbose_eval=False, early_stopping_rounds=50)
    
    length = len(evals_result['training']['rmse'])
    sum_rmse_min = float("inf")
    best_iter = -1
    for i in range(length):
        sum_rmse = evals_result['training']['rmse'][i]+evals_result['validating']['rmse'][i]
        if sum_rmse< sum_rmse_min:
            sum_rmse_min = sum_rmse
            best_iter = i
    return sum_rmse_min

# Return the trained model
def get_trained_model(x,**kwargs):
    
    param = set_param(x)
    
    data_train = kwargs['data_train']
    data_validate = kwargs['data_validate'] 
    

    evallist = [(data_train, 'training'),(data_validate, 'validating')]
    evals_result = {}
    
    xgb_model = xgb.train(param, data_train, num_boost_round=500, evals=evallist, evals_result=evals_result, early_stopping_rounds=50)
    
    
    return xgb_model, evals_result
