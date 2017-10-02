# import the necessary packages
import numpy as np
import os
import glob
import cv2.cv as cv
import cv2

class video:
    def __init__(self,path):
        global newpath
        self.numberOfSamples = 20
        self.requiredMatches = 2
        self.distanceThreshold = 20
        self.fname=[]
        self.path=path
        newpath = r'Frames'
        if not os.path.exists(newpath): os.makedirs(newpath)
        newpath = r'NewFrames'
        if not os.path.exists(newpath): os.makedirs(newpath)
        bigSampleArray = self.initialFraming(self.path)
        self.processVideo(bigSampleArray)

    def sort_files(self):
        for file in sorted(glob.glob("Frames/*.*")):
            s=file.split('/')
            a=s[-1].split('\\')
            x=a[-1].split('.')
            self.fname.append(int(x[0]))
        return(sorted(self.fname)) 

    def initialFraming(self,path):
        global cap
        global success
        sampleIndex=0
        cap = cv2.VideoCapture(path)
        success,frame=cap.read(cv.CV_IMWRITE_JPEG_QUALITY)       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        height,width = gray.shape[:2]

        samples = np.array([[0 for x in range(0,self.numberOfSamples)] for x in range(0,(height*width))])

        tempArray = np.reshape(gray,(height*width)).T
        
        samples[:,sampleIndex]= np.copy(tempArray)
        sampleIndex+=1

        while (success and sampleIndex!=(self.numberOfSamples)):
            success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            tempArray = (np.reshape(gray,(height*width))).T
            samples[:,sampleIndex]= np.copy(tempArray)
            sampleIndex+=1

        return samples
            
    def processVideo(self,bigSampleArray):
        global success
        global cap
        
        samples= bigSampleArray

        i=0
        while success:
            success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            height,width = gray.shape[:2]
            tempArray = np.reshape(gray,(height*width)).T
            segmentationMap = np.copy(tempArray)*0
            for p in range(0,len(bigSampleArray)):
##                print "Value of p is: ",p
                count = index = distance = 0

                while((count < self.requiredMatches) and (index < self.numberOfSamples)):
                    distance = np.linalg.norm(tempArray[p]-samples[p][index])
##                    print "Euclidean distance is: ",distance
                    if (distance < self.distanceThreshold):
                        count += 1
##                        print "count reached" ,count
                    index += 1

                if(count<self.requiredMatches):
                    segmentationMap[p]=255
                else:
                    segmentationMap[p]=0
            segmentationMap= np.reshape(segmentationMap,(height,width))
            NewPath="NewFrames/"+ str(i+1) + ".jpg"
            cv2.imwrite(NewPath,segmentationMap)
            i+=1

        cv2.destroyAllWindows()    
##        global array
##        x=np.array([])
##        mean=0
##        for i in range(0,len(array)):
##            pic= "Frames/" + str(i+1) + ".jpg"
##            frame=cv2.imread(pic)
##            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##            gray = cv2.GaussianBlur(gray, (21, 21), 0)
##            height,width = gray.shape[:2]
##            newImage=np.array([[0 for x in range(0,width)]for x in range(0,height)])
##            for j in range(0,height):
##                for k in range(0,width):
##
##                    if(i==0):
##                       continue
##                    else:
##                        if(gray[j][k]<=mean):
####                            print "Background",
##                            newImage[j][k]=0
##                        else:
####                            print "Foreground",
##                            newImage[j][k]=255
##
##                    x=np.append(x,int(gray[j][k]))    
##                    mean=np.mean(x)
####                    print mean
##                    
##            NewPath="NewFrames/"+ str(i+1) + ".jpg"
##            cv2.imwrite(NewPath,newImage)   
####          print np.mean(x),
####          print np.array(x.append(int(gray[0][0]))).mean()
            
def main():
    path_file='walking.avi'
    v = video(path_file)
    
main()            
        
'''    
    def framing(self,path):
        global newpath
        cap = cv2.VideoCapture(path)
        success,frame=cap.read(cv.CV_IMWRITE_JPEG_QUALITY)
              
        count = 1;
        
        firstFrame = None
# loop over the frames of the video
        while success:
	# resize the frame, convert it to grayscale, and blur it
##            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (51, 51), 0)

##            print newpath, len(os.listdir(newpath))
##            print len([name for name in os.listdir(newpath) if os.path.isfile(name)])

	# if the first frame is None, initialize it
            if firstFrame is None:
                   firstFrame = gray
                   continue

	# compute the absolute difference between the current frame and
	# first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
##            print len(cnts)
	# loop over the contours
            for c in cnts:
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
##		if(len(c) > 0):
##                    m= np.mean(c[0],axis=0)
##                    measuredTrack[count-1,:]=m[0]
##                    plt.plot(m[0,0],m[0,1],'ob')
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##                cv2.drawContours(frame, c, -1, (0,255,0), 3)
                 
	# show the frame and record if the user presses a key
            cv2.imshow("Feed", frame)
            cv2.imshow("Thresh", thresh)
##            cv2.imshow("Frame Delta", frameDelta)
            key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
            if key == ord("q"):
                    break
                
            cv2.imwrite("Frames/%d.jpg" % count, frame)           # save frame as JPEG file
            count += 1
            success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)

        array=self.sort_files()
        print array
        cv2.destroyAllWindows()
        cap.release()
        plt.show()
'''


 

