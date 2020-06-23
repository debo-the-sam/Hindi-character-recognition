import matplotlib.pyplot as plt

def plot(X,Y,height,width):
    z = X.reshape(height,width)
    plt.imshow(z,cmap = plt.cm.binary)
    plt.show()
    print("Class label: {}".format(Y))


def predPlot(X,height,width):
    z = X.reshape(height,width)
    plt.imshow(z,cmap = plt.cm.binary)
    plt.show()