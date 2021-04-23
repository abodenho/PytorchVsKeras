def writeIntoFiles(lstValTest, lstValTrain,time, name):

    y = open("Timer.txt","a")
    y.write("{name} : {time} \n".format(name=name,time=time))
    y.close()
    f = open("./{name}.txt".format(name=name), "w")
    toWrite = "Epoch,ACC_Test,ACC_Train \n"
    f.write(toWrite)
    for i in range(len(lstValTest)):
        precisionTest = lstValTest[i]
        precisionEntrainement = lstValTrain[i]
        toWrite = str(i+1) +","+str(round(precisionTest,4))+ ","+str(round(precisionEntrainement,4))+" \n"
        f.write(toWrite)
    f.close()

