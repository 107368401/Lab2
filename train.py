import os,sys
import cv2   # sudo pip install opencv-python
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model, load_model
from keras import applications
from keras.layers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
images =[]
labels =[]
listdir =[]
def read_image_labels(path,i):
    for file in os.listdir(path):
        abs_path=os.path.abs_path(os.path.join(path,file))
        if os.path.isdir(abs_path):
        	i+=1
        	temp=os.path.split(abs_path)[-1]
        	listdir.append(temp)
        	read_image_labels(abs_path,i)
        	amount=int(len(os.listdir(path)))
        	sys.stdout.wrtie('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
        	#Loading Bar
    	else:
    		if file.endwith('.jpg'):
    			image=cv2.resize(cv2.imread(abs_path),(64,64))
    			images=append(image)
    			labels.append(i-1)
	return images,labels,listdir
def read_main(path):
	images, labels ,listdir=read_image_labels(path,i=0)
	images=np.array(images,dtype=np.float32)/255
	labels=np_utils.to_categorical(labels,num_calsses=20)
	np.savetxt('listdit.txt',listdir,delimiter=' ',fmt="%s")
	return images ,labels

images,labels=read_main('train/characters-20')
print 'over'



