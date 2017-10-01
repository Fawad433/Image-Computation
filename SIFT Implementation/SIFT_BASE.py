#Final Base Program
import cv2
import numpy as np
import operator

#Intialisation
sift = cv2.SIFT(0,3,0.08,10,1.6)
frame_overlap=np.zeros((30,3),dtype=float)
ptr=1
count=0
cap=cv2.VideoCapture('example1.mov')
while(cap.isOpened()):
        ret,frame=cap.read()
        
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        (kp,descs)=sift.detectAndCompute(gray,None)
        length=len(kp)
        print("# kps: {}, descriptors: {}".format(length, descs.shape))
        frame=cv2.drawKeypoints(frame,kp,(255,0,0),4)
      
        #Frame Operation
        frame_list=np.zeros((length,1),dtype=float)
        
        for i in range (0,length):
                frame_list[i,:]=kp[i].response
                
        index, value = max(enumerate(frame_list), key=operator.itemgetter(1)) 
        index=int(index)
        x=int(kp[index].pt[0])
        y=int(kp[index].pt[1])
        #Check For Frame Overlap
        while(ptr==1):
                for i in range (0,29):
                        x_cmp=int(frame_overlap[i,0])
                        y_cmp=int(frame_overlap[i,1])
                        radius=int(frame_overlap[i,2])
                        if (x > (x_cmp-radius)) and (x < (x_cmp+radius)):
                                ptr=0
                        if (y > (y_cmp-radius)) and (y < (y_cmp+radius)):
                                ptr=0
                
                if ptr==1:
                        ptr=0
                        
                else: 
                        ptr=1
                        index=index+1
                        if (index >= (length-1)):
                                print("Correct Attention window cannot be extracted from this frame due to severe overlap!! Extracting attention window with Highest Response")
                                ptr=0
                                index=0        
                        x=int(kp[index].pt[0])
                        y=int(kp[index].pt[1])
        
        ptr=1                              
        val=int((kp[index].size)/2) 
        #Store Frame Data
        frame_overlap[count,0]=x
        frame_overlap[count,1]=y
        frame_overlap[count,2]=val
        if count==29:
                count=0           
        else:
                count+=1
                               
        cv2.rectangle(frame,(x-val,y-val),(x+val,y+val),(255,0,0),3)
        cv2.imshow('Output',frame)
        
        if cv2.waitKey(15) == 13:
                break
                
cap.release()
cv2.destroyAllWindows()