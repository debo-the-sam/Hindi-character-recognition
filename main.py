import pandas as pd
from HCR.preprocess import prepareInputData
from HCR.model import DevnagiriCNN
from HCR.visualize import plot

train_data_path = r'Dataset/Devnagiri/train.csv'
test_data_path = r'Dataset/Devnagiri/test.csv'
model_path = r"SavedModel/devanagiri.h5"
     
def main():
    ### Load Dataset ###
    train_data = pd.read_csv(train_data_path,header=None)
    test_data = pd.read_csv(test_data_path,header=None)

    ### Prepare Dataset in the form of X and Y values

    """Training data"""
    X_train,Y_train = prepareInputData(train_data)

    """Testing data"""
    X_test,Y_test = prepareInputData(test_data)

    ### Parameters
    height = 32
    width = 32
    channels = 1
    num_classes = 10

    ### Input and output shape
    # print(X_train.shape,Y_train.shape)

    ### Visualize each item 
    # plot(X_train[0],Y_train[0],height,width)


    """Devnagiri CNN model"""
    ### initialize model pipeline
    model = DevnagiriCNN()

    ### initialize model pipeline
    # model.initModel(height = height, width = width, channels = channels,classes = num_classes)

    ### Train model 
    # model.train(X= X_train,Y = Y_train,epochs=20,test_size = 0.1,batch_size=64,model_save = True)

    ### Load CNN model
    model.loadModel(height = height, width = width, channels = channels,classes = num_classes,path = model_path)

    ### Validate model (Don't run unless you have 8GB GPU RAM or 16 GB CPU RAM)
    model.validate(X = X_test,Y = Y_test)

    ### Predict Model 
    # model.predict(X = X_test)



if __name__ == '__main__':
    main()
