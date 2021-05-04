def writeIntoFilesKeras(history,time, name):
    y = open("Timer.txt","a")
    y.write("{name} : {time} \n".format(name=name,time=time))
    y.close()
    f = open("./{name}.txt".format(name=name), "w")
    toWrite = "Epoch,ACC_Test,ACC_Train \n"
    f.write(toWrite)
    for i in range(len(history.history['val_acc'])):
        toWrite = "{epoch},{ACC_test},{ACC_train}".format(epoch = i+1,ACC_train = _arrondi(history.history['acc'][i]),ACC_test = _arrondi(history.history['val_acc'][i]))
        f.write(toWrite)
    f.close()

def _arrondi(number):
    return round(number,4)