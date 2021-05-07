from commun.CONSTANTE import *
import torch.nn as nn
import torch.optim as optim

def getLossFunction():
    if FUNCTION_LOSS == "binary_crossentropy":
        function = nn.BCELoss()
    elif FUNCTION_LOSS == "mean_squared_error":
        function = nn.MSELoss()
    elif FUNCTION_LOSS == "mean_absolute_error":
        function = nn.L1Loss()
    else:
        raise Exception

    return function


def getOptimize(model):
    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters())
    elif OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(),lr=0.01)
    elif OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise Exception

    return optimizer