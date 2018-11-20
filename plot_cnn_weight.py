import cv2
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import load_model

model=load_model('100_256.h5')

def plot_filters(layer,x,y):
    
    filters=layer.get_weights()

    print filters
    fig=plt.figure()
    for j in range(len(filters)):
        ax=fig.add_subplot(y,x,j+1)


        ax.matshow(filters[j][0],cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
   # plt.show()
    return plt


print model.layers[0].filters



x1w = model.layers[0].get_weights()[0][:,:,0,:]
for i in range(1,64):
    plt.subplot(8,8,i)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
plt.show()



x1w = model.layers[3].get_weights()[0][:,:,0,:]
for i in range(1,128):
    plt.subplot(10,13,i)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
plt.show()


x1w = model.layers[4].get_weights()[0][:,:,0,:]
for i in range(1,128):
    plt.subplot(10,13,i)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
plt.show()

x1w = model.layers[7].get_weights()[0][:,:,0,:]
for i in range(1,256):
    plt.subplot(16,16,i)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
plt.show()

