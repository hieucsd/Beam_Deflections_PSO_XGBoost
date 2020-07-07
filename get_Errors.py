# function to get precision and call of binary classification with labels 0 or 1
# Y is the vector of true labels
# predicted_Y is the predicted labels
import numpy as np
def get_MAPE(Y,predicted_Y):
    y_true, y_pred = np.array(Y), np.array(predicted_Y)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
