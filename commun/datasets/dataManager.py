import os
import pandas as pd
from commun.CONSTANTE import *
from math import *

class DataManager:
    def __init__(self, path):

        if "csv" in path:
            self.isPicture = False
            self.df = pd.read_csv(path)
            self._splitData()

        else:
            self.isPicture = True
            self.path = path
            self._comptageTot()

    def _splitData(self):
        self.numberRowTest = self.getNumberTest()


    def getNumberTest(self):
        if self.isPicture:
            rep = self._getNumberTestPicture()
        else:
            rep = self._getNumberTestCSV()

        return rep

    def _getNumberTestPicture(self):
        return self.numberTestPicture

    def _getNumberTestCSV(self):
        numberRows = len(self.df.axes[0])
        return ceil(numberRows * POURCANT_DATA_TEST)

    def getNumberTrain(self):
        if self.isPicture:
            rep = self._getNumberTrainPicture()
        else:
            rep = self._getNumberTrainCSV()

        return rep

    def _getNumberTrainPicture(self):
        return self.numberTrainPicture

    def _getNumberTrainCSV(self):
        return len(self.df.axes[0])



    def getPathTrainPicture(self):
        return self.path + "/" + "training_set"

    def getPathTestPicture(self):
        return self.path + "/" + "test_set"

    def getTrainCSV(self):
        return (self.df.iloc[self.numberRowTest:,:-1].values,self.df.iloc[self.numberRowTest:,-1].values)

    def getTestCSV(self):
        return (self.df.iloc[:self.numberRowTest, :-1].values, self.df.iloc[:self.numberRowTest, -1].values)

    def _comptageTot(self):
        """ compte le nombre total de photo """
        imageTest = 0
        imageTrain = 0
        foldersTest = os.listdir(self.getPathTestPicture())
        foldersTrain = os.listdir(self.getPathTrainPicture())
        for subFolders in foldersTrain:
            imageTrain += self._comptageImage(self.getPathTrainPicture() + "/" + subFolders)
        for subFolders in foldersTest:
            imageTest += self._comptageImage(self.getPathTestPicture() + "/" + subFolders)

        self.numberTestPicture = imageTest
        self.numberTrainPicture = imageTrain

    def _comptageImage(self,pathFolder):
        folder = os.listdir(pathFolder)
        return len(folder)



