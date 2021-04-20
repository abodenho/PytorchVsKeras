def writeIntoFile(lstValTest, lstValTrain, path2write):
    f = open(path2write, "a")
    toWrite = "Epoch,ACC_Test,ACC_Train,Diff_ACC \n"
    f.write(toWrite)
    for i in range(len(lstValTest)):
        precisionTest = lstValTest[i]
        precisionEntrainement = lstValTrain[i]
        toWrite = str(i+1) +","+str(round(precisionTest,4))+ ","+str(round(precisionEntrainement,4))+ "," + str(round(abs(precisionTest-precisionEntrainement),4)) +  "\n"
        f.write(toWrite)
    f.close()