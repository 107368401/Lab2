import cv2
import numpy as np
from keras.models import load_model

def read_images(path):
	images=[]
	for i in range(990):
		image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(64,64))
		images.append(image)
	images=np.array(images,dtype=np.float32)/255
	return images
images=read_images('test/')

def transform(listdir,label,lenSIZE):
	label_str=[]
	for i in range(lenSIZE):
		temp=listdir[label[i]]
		label_str.append(temp)
	return label_str
model=load_model('100_256.h5')
#model=load_model('1_1.h5')
predict=model.predict_classes(images,verbose=1)
label_str=transform(np.loadtxt('listdir.txt',dtype='str'),predict,images.shape[0])

np.savetxt('test_sorce.csv',label_str,delimiter=',',fmt="%s")
