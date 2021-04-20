import pandas as pd
import numpy as np



class CSVTraitement:

    def __init__(self,csvToLoad):
        self.df = pd.read_csv(csvToLoad)

    def deletColumn(self,lstColumn):
        self.df = self.df.drop(lstColumn, axis=1)

    def getData(self):
        return self.df

    def oneHotEncoder(self,column):
        tmp = pd.get_dummies(self.df[column],drop_first=True) #drop first pour enlevé une colonne
        indexCol = self.df.columns.get_loc(column)
        self.df = self.concatenateColumn(self.concatenateColumn(self.df.iloc[:,:indexCol],tmp),self.df.iloc[:,indexCol+1:])


    def concatenateColumn(self,elemOne,elemTwo):
        return pd.concat([elemOne,elemTwo],axis=1)


    def normalize(self,column):
        self.df[column] = self.df[column] / self.df[column].abs().max()

    def saveData(self,path):
        self.df.to_csv(path,index=False)

    def deleteNameColum(self):
        self.df = self.df.iloc[:,:]


data = CSVTraitement("Churn_Modelling.csv")

data.deletColumn(["RowNumber","CustomerId","Surname"])

data.oneHotEncoder("Geography")
data.oneHotEncoder("Gender")

data.normalize("CreditScore")
data.normalize("Age")
data.normalize("Tenure")
data.normalize("Balance")
data.normalize("NumOfProducts")
data.normalize("EstimatedSalary")

data.deleteNameColum()

data.saveData('Churn_Modelling_traited.csv')