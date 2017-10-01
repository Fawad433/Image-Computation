import cv2
import numpy as np
import array 
import operator
import os

dirname='Output_Frame'
dirname_1='Attention_Window'
#Kindly uncomment the below 2 lines when executing the program for the FIRST TIME ONLY!!
#os.mkdir(dirname)
#os.mkdir(dirname_1)

#Intialisation
#video_name=input('Enter the name of the input video:')
surf=cv2.xfeatures2d.SURF_create(1000,5,5) #Hessian Threshold
frame_overlap=np.zeros((30,3),dtype=float)
ptr=1
count=0
frame_count=0

cap=cv2.VideoCapture('example1.mov')

while(cap.isOpened()):
        ret,frame=cap.read()
        
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''
        #Sobel Edge
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)  # X-Coordinate
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)  # Y-Coordinate
        sobel_x = cv2.convertScaleAbs(sobelx)   # converting back to uint8
        sobel_y = cv2.convertScaleAbs(sobely)
        gray = cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
        
        '''
        '''
        #Laplacian 
        laplacian = cv2.Laplacian(gray,cv2.CV_64F)
        gray = cv2.convertScaleAbs(laplacian) 
        ''' 
        #SURF
        (kp,descs)=surf.detectAndCompute(gray, None)
        length=len(kp)
        print("#Frame Number:{}, Keypoints: {}, Descriptors: {}".format(frame_count, length, descs.shape))
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
                                print("Attention window cannot be extracted from this frame due to severe overlap!! Extracting attention window with Highest Response")
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
        
        attention_window = frame[y-val:y+val , x-val:x+val]                       
        cv2.rectangle(frame,(x-val,y-val),(x+val,y+val),(255,0,0),3)
        cv2.imshow('Output',frame)
        cv2.imwrite(os.path.join(dirname, "%d.png" %(frame_count)),frame)
        cv2.imwrite(os.path.join(dirname_1, "%d.png" %(frame_count)),attention_window)
        frame_count+=1
        
        if cv2.waitKey(15) == 13:
                break
                
cap.release()
cv2.destroyallwindows()
