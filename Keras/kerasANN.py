
import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from commun.CommunFonction import *
from commun.datasets.dataManager import *
from commun.CONSTANTE import *
import sys

OPTIMIZER = sys.argv[1]
FUNCTION_LOSS = sys.argv[2]
NUMBER_NEURONE_HIDDEN_LAYER_ANN = int(sys.argv[3])
PathDataSet = sys.argv[4]


beginTime = time.time()
data = DataManager(PathDataSet)

xTrain, yTrain = data.getTrainCSV()
xTest, yTest = data.getTestCSV()



def buildANN():

    model = Sequential()
    model.add(Dense(NUMBER_NEURONE_HIDDEN_LAYER_ANN,activation=FUNCTION_ACTIVATION,input_shape=(ANN_NUMBER_INPUT,)))
    for j in range(NUMBER_HIDDEN_LAYER-1):
        model.add(Dense(units=NUMBER_NEURONE_HIDDEN_LAYER_ANN, activation=FUNCTION_ACTIVATION))

    model.add(Dense(NUMBER_NEURONE_OUTPUT,activation=FUNCTION_OUTPOUT))
    model.compile(optimizer=OPTIMIZER,loss=FUNCTION_LOSS,metrics=METRIC)
    return model

ANN = buildANN()

saves = ANN.fit(xTrain,
                yTrain,
                epochs=NUMBER_EPOCH,
                batch_size=BATCH_SIZE,
                validation_data=(xTest,yTest),
                verbose=0)

endTime = time.time() - beginTime

writeIntoFilesKeras(saves.history["acc"],saves.history["val_acc"],saves.history["loss"],saves.history["val_loss"],endTime,
                    f"keras_{OPTIMIZER}_{NUMBER_NEURONE_HIDDEN_LAYER_ANN}_{FUNCTION_LOSS}")

