import os
lstOptimizer = ["adam", "sgd", "rmsprop"]
lstLossFunction = ["binary_crossentropy", "mean_squared_error", "mean_absolute_error"]
lstNeuroneHiddenLayer = [8, 16, 32]

PathDataSet = "../../commun/datasets/ANN/Churn_Modelling_traited.csv"

for OPTIMIZE in lstOptimizer:
    for FUNCTION_LOSS in lstLossFunction:
        for NUMBER_NEURONE_HIDDEN_LAYER_ANN in lstNeuroneHiddenLayer:
            os.system(f"python ../../Pytorch/pyANN.py {OPTIMIZE} {FUNCTION_LOSS} {NUMBER_NEURONE_HIDDEN_LAYER_ANN} {PathDataSet}")
            os.system(f"python ../../Keras/kerasANN.py {OPTIMIZE} {FUNCTION_LOSS} {NUMBER_NEURONE_HIDDEN_LAYER_ANN} {PathDataSet}")

