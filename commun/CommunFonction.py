def writeIntoFilesKeras(lstVal,lstValACC,lstLOSS,lstLOSSACC,time, name):
    y = open("Timer.txt","a")
    y.write("{name} : {time} \n".format(name=name,time=time))
    y.close()
    f = open("./{name}.txt".format(name=name), "w")
    toWrite = "Epoch,ACC_Train,LOSS_Train,ACC_Test,LOSS_test \n"
    f.write(toWrite)
    for i in range(len(lstVal)):
        toWrite = "{epoch},{ACC_train},{LOSS_train},{ACC_test},{LOSS_test} \n".format(epoch = i+1, ACC_train = arrondi(lstVal[i]), ACC_test = arrondi(lstValACC[i]),
                                                                                      LOSS_test = arrondi(lstLOSSACC[i]), LOSS_train= arrondi(lstLOSS[i]))
        f.write(toWrite)
    f.close()

def arrondi(number):
    return round(number,4)