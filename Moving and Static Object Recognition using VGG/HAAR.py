import cv2
import numpy as np
import keras
import tensorflow as tf
import vgg16 as vg

from scipy.misc import imread, imresize
from imagenet_classes import class_names

video_src = 'example1.mov'
cascade_src = 'car.xml'

sess=tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
vgg = vg.vgg16(images, 'vgg16_weights.npz', sess)

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while (cap.isOpened()):
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
    	i=1
	if ((w*h)<5000):
		continue
	else:
        	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)  
        	car= img[y:y+h,x:x+w]   
        	i+=1
    
    vgg1 = vg.imresize(car, (224, 224))
    #vgg2 = vg.imresize(car, (224, 224))
    image_stack  = np.stack([vgg1])
    probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})
    preds = np.argmax(probs, axis=1)
    for index, p in enumerate(preds):
    	cv2.putText(img,"{}".format(class_names[p]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    for index, p in enumerate(preds):
    	print("Prediction: %s; Probability: %f"%(class_names[p], probs[index, p]))  
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
