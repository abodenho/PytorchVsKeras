import os

def comptageTot(folderPathTrain, folderPathTest):
    """ compte le nombre total de photo """
    imageTest = 0
    imageTrain = 0
    foldersTest = os.listdir(folderPathTest)
    foldersTrain = os.listdir(folderPathTrain)
    for subFolders in foldersTrain:
        imageTrain += comptageImage(folderPathTrain + "/" + subFolders)
    for subFolders in foldersTest:
        imageTest += comptageImage(folderPathTest + "/" + subFolders)
    return (imageTrain, imageTest)


def comptageImage(pathFolder):
    folder = os.listdir(pathFolder)
    return len(folder)

def writeIntoFile(lstValTest, lstValTrain, name):
    f = open("./{name}.txt".format(name=name), "a")
    toWrite = "Epoch,ACC_Test,ACC_Train \n"
    f.write(toWrite)
    for i in range(len(lstValTest)):
        precisionTest = lstValTest[i]
        precisionEntrainement = lstValTrain[i]
        toWrite = str(i+1) +","+str(round(precisionTest,4))+ ","+str(round(precisionEntrainement,4))+" \n"
        f.write(toWrite)
    f.close()

