##LAYER
NUMBER_HIDDEN_LAYER = 3
NUMBER_LAYER_POOL_CONV = 4

SIZE_MATRICE_POOLING = (2,2)

##NEURONE
NUMBER_NEURONE_HIDDEN_LAYER_CNN = 128
NUMBER_NEURONE_HIDDEN_LAYER_ANN = 16
NUMBER_NEURONE_OUTPUT = 1

##FUNCTION
FUNCTION_ACTIVATION = "relu"
FUNCTION_OUTPOUT = "sigmoid"
OPTIMIZER = 'adam'
FUNCTION_LOSS = "binary_crossentropy"
BATCH_SIZE = 32

##Mode
CLASS_MODE = "binary"


##IMAGE
IMAGE_SIZE = 150
IMAGE_SIZE_TUPLE = (IMAGE_SIZE,IMAGE_SIZE)

##OTHER
NUMBER_EPOCH = 2
NUMBER_FILTRE = 32
STRIDE_POOLING = 2
STRIDE_CONVOLUTION = 1
KERNEL_SIZE = 3

##KERAS
METRIC = ["acc"]


##DATA MANAGER
POURCANT_DATA_TEST = 0.2
POURCANT_DATA_TRAIN = 1 - POURCANT_DATA_TEST
