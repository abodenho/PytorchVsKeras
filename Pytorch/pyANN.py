from Pytorch.constanteChooserValue import *
import time
import torch.nn as nn
from commun.CONSTANTE import *
import torch
from commun.CommunFonction import writeIntoFilesKeras
from commun.datasets.dataManager import DataManager
import torch.optim as optim
from math import ceil


beginTime = time.time()
data = DataManager("../commun/datasets/ANN/Churn_Modelling_traited.csv")


trainTensor,trainLabelTensor = data.getTrainCSV()
testTensor, testLabelTensor = data.getTestCSV()



trainTensor,trainLabelTensor = torch.tensor(trainTensor).float(),torch.tensor(trainLabelTensor).unsqueeze(1).float()
testTensor, testLabelTensor  = torch.tensor(testTensor).float(),torch.tensor(testLabelTensor).unsqueeze(1).float()
#unsqueeze(1) => augmente d'une dimmension | ixj => nxixj (n , i, j représentant des dimmension)



def splitBatch(tensorToSplit,numberBatch):
    lstBatch = []
    sizeBatch = len(tensorToSplit)/numberBatch
    for i in range(numberBatch):
        supBorne = ceil(sizeBatch*(i+1))
        infBorne = ceil(sizeBatch*i)
        lstBatch.append(tensorToSplit[infBorne:supBorne][:])

    return lstBatch

trainBatch = splitBatch(trainTensor, BATCH_SIZE)
trainBatchLabel = splitBatch(trainLabelTensor,BATCH_SIZE)
testBatch = splitBatch(testTensor,BATCH_SIZE)
testBatchLabel = splitBatch(testLabelTensor,BATCH_SIZE)


model = nn.Sequential(
    nn.Linear(ANN_NUMBER_INPUT,NUMBER_NEURONE_HIDDEN_LAYER_ANN),
    nn.ReLU(),
    nn.Linear(NUMBER_NEURONE_HIDDEN_LAYER_ANN,NUMBER_NEURONE_HIDDEN_LAYER_ANN),
    nn.ReLU(),
    nn.Linear(NUMBER_NEURONE_HIDDEN_LAYER_ANN,NUMBER_NEURONE_HIDDEN_LAYER_ANN),
    nn.ReLU(),
    nn.Linear(NUMBER_NEURONE_HIDDEN_LAYER_ANN,NUMBER_NEURONE_OUTPUT),
    nn.Sigmoid()
)

optimizer = optim.Adam(model.parameters())


def calculateAccuracy(result,target):
    goodPredictionTensor = (result > THRESHOLD_TRUE) == target
    total = len(goodPredictionTensor)
    goodPrediction = 0
    for booleanValue in goodPredictionTensor:
        if booleanValue:
            goodPrediction +=1
    return goodPrediction/total

def calculateMean(lstValue):
    total = 0
    for valueAcc in lstValue:
        total += valueAcc
    return total/len(lstValue)

lstAcc = []
lstValAcc = []
lstLoss = []
lstValLoss = []

def trainLoop(numberEpoch,optimizer,model,lossFunction,
              trainBatch,trainBatchLabel,
              testBatch,testBatchLabel):

    for epoch in range(1,numberEpoch+1):
        subLstAcc = []
        subLstAccVal = []
        subLstLoss = []
        subLstLossVal = []
        for train,trainLabel,test,testLabel in zip(trainBatch,trainBatchLabel,
                                                   testBatch,testBatchLabel):
            resultTrain = model(train)
            subLstAcc.append(calculateAccuracy(resultTrain,trainLabel))
            lossTrain = lossFunction(resultTrain,trainLabel)
            subLstLoss.append(lossTrain.item())
            with torch.no_grad(): #désactive l'accumulation pour mise a jour des poids et biais
                resultTest = model(test)
                subLstAccVal.append(calculateAccuracy(resultTest,testLabel))
                lossTest = lossFunction(resultTest,testLabel)
                subLstLossVal.append(lossTest.item())

            optimizer.zero_grad() #met a jour le buffer servant a l'accumulation
            # pour la mise a jour poids et biais(époque précédente)
            lossTrain.backward() #calcule le gradient des donnée
            optimizer.step() #met a jours les données

        lstAcc.append(calculateMean(subLstAcc))
        lstValAcc.append(calculateMean(subLstAccVal))
        lstLoss.append(calculateMean(subLstLoss))
        lstValLoss.append(calculateMean(subLstLossVal))



lossFunction = getLossFunction()

trainLoop(NUMBER_EPOCH,optimizer,model,lossFunction,
          trainBatch,trainBatchLabel,
          testBatch, testBatchLabel)

endTime = time.time() - beginTime


writeIntoFilesKeras(lstAcc,lstValAcc,lstLoss,lstValLoss,endTime,
                    f"pytorch_{OPTIMIZER}_{NUMBER_NEURONE_HIDDEN_LAYER_ANN}_{FUNCTION_LOSS}")

