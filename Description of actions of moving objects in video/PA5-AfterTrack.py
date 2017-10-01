from __future__ import print_function

import sys
import vgg16 as vg
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import tensorflow as tf
import operator
import array
import random
import heapq
PY3 = sys.version_info[0] == 3

if PY3:

    xrange = range

import numpy as np
import cv2
import csv
from heapq import nlargest
from operator import itemgetter

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)


def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5


class MOSSE:
    def __init__(self, frame, rect,count,count1):
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.count = count
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))
        self.count1 = count1       
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()
        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)


    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            return

        self.pos = x+dx, y+dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)
        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis


    def draw_state(self, vis,vgg,images,sess, count1,frame_count,datadynamic,x_new,x_old,xs,static_label):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
            checker = 0
        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))           
            checker = 100
        
        x1a = x1
        y1a = y1
        cropped = vis[y1:y1+y2 , x1:x1+x2]
        img = cropped
        x1, y1, z = img.shape
        if x1 != 0 and y1 != 0    :
            if(x1>y1):
                    
                x1 = (x1*224)/y1
                y1 = 224
                img_res = cv2.resize(img, (y1,int(x1)), interpolation = cv2.INTER_CUBIC)
    
                x1= x1/2;
                x1 = int (x1)
                img_res = img_res[(x1)-112:(x1)+112, 0:224]
    
                    
            else:
                    
                y1 = int((y1*224)/x1)
                x1 = 224
                img_res = cv2.resize(img, (y1,x1), interpolation = cv2.INTER_CUBIC)
                    
                y1 = y1/2;
                    
                y1 = int (y1)
                Y1 = y1 - 122
                Y2 = y1 + 122
    
                img_res = img_res[0:224 , Y1:Y2]
                
            vgg1 = vg.imresize(img_res, (224, 224))
            image_stack  = np.stack([vgg1])
            probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})
            preds = np.argmax(probs, axis=1)
    
            for index, p in enumerate(preds):
                cv2.putText(vis,"{}".format(class_names[p]),(x1a,y1a-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)                
            text = str(self.count1) 
            cv2.putText(vis,"{}".format(text),(x1a,y1a),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
#            print("Dynamic,Frame Number:{}, Track Number:{},X-Axis:{}, Y-Axis:{}, Dynamic Prediction:{}, Probablity:{}".format(frame_count,text,x1,y1,class_names[p],probs[index,p]))
            datadynamic=np.vstack((datadynamic,np.array(("Dynamic",frame_count,text,x1,y1,class_names[p],probs[index,p]))))
#            x_new[self.count1]=x
#            if (x_new[self.count1]>x_old[self.count1]):
#            	
#            	print("{} is moving towards the right".format(class_names[p]))
#            	if (xs>x_new[self.count1]):
#            		print("Track{} is moving towards {}".format(text, static_label))
#            	else:
#            		print("Track{} is moving away {}".format(text, static_label))	
#            else:
#            	print("{} is moving towards the left".format(class_names[p]))
#            	if (xs<x_new[self.count1]):
#            		print("Track{} is moving towards {}".format(text, static_label))
#            	else:
#            		print("Track{} is moving away {}".format(text, static_label))
            
            x_old[self.count1]=x
        return checker,datadynamic     


    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win



    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr


    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1

class App:    

    def __init__(self, video_src, paused = False):
       
        def run(self,vgg,images,sess,frame,cnts, count1,frame_count):
            checker = 0
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            while True:                
                if not self.paused:                   
                    ret, self.frame = self.cap.read()    
                    if not ret:    
                        break
                    frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)   
                    for tracker in self.trackers:   
                        tracker.update(frame_gray)
   
                vis = self.frame.copy()
      
                (kp,descs)=surf.detectAndCompute(gray, None)
                length=len(kp)
                frame_list=np.zeros((length,1),dtype=float)
                for i in range (0,length):
                    frame_list[i,:]=kp[i].response
               
                index=random.randint(0,length-1)
                xs=int(kp[index].pt[0])
                ys=int(kp[index].pt[1])                      
                val=int((kp[index].size)/2) 
                new_val=val
                #Resize Attention Window
                if val<50:
                    if xs<100 or ys<100 or xs>(width-100) or ys>(height-100):
                        new_val=val 
                    else:
                        new_val=100
                attention_window = vis[ys-new_val:ys+new_val , xs-new_val:xs+new_val]

                if len(self.trackers) > 0:   
                    cv2.imshow('tracker state', self.trackers[-1].state_vis)               
                x0,y0,x1,y1 = self.rect  
                cv2.rectangle(vis,(xs-new_val,ys-new_val),(xs+new_val,ys+new_val),(255,0,0),3)
                vgg1 = vg.imresize(attention_window, (224, 224))
                image_stack  = np.stack([vgg1])
                probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})
                preds = np.argmax(probs, axis=1)
                for index, p in enumerate(preds):
                    cv2.putText(vis,"{}".format(class_names[p]),(xs-new_val,ys-new_val),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

                #print("Still,Frame Number:{},Frame Number:{},X-Axis:{}, Y-Axis:{}, Static Prediction:{}, Probablity:{}".format(frame_count,frame_count,xs,ys,class_names[p],probs[index,p]))
                self.datastatic=np.vstack((self.datastatic,np.array(("Still",frame_count,frame_count,xs,ys,class_names[p],probs[index,p]))))
                static_label=class_names[p]
                for tracker in self.trackers:  
                    (x, y), (w, h) = tracker.pos, tracker.size
                    x11, y11, x12, y12 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
                    c,self.datadynamic =  tracker.draw_state(vis,vgg,images,sess, count1,frame_count,self.datadynamic,x_new,x_old,xs,static_label)
                    
                    if c == 100:
                        checker = 100  
                cv2.imshow('frame', vis)
                fvid.write(vis)   
                ch = cv2.waitKey(10)                     
                break               
                if ch == ord(' '):                    
                    self.paused = not self.paused    
                if ch == ord('c'):
                    self.trackers = []
        
            return checker        
        #Initialisation
        global fvid,lengthvid,frame_count
        self.datastatic = []
        self.datadynamic = []
        x_new=np.zeros((10,1))
        x_old=np.zeros((10,1))
        fvid = cv2.VideoWriter('video.avi',1,24.0,(1280,720))
        p = 0
        k = 0
        count = 0
        cnts = [] 
        cnt1 = []
        self.datastatic.append(("Type","Frame Number","Frame Number","X-Axis","Y-Axis","Object Label","Probablity"))
        self.datadynamic.append(("Type","Frame Number","Track Number","X-Axis","Y-Axis","Object Label","Probablity"))
        frame_count=0
        count1 = 0
        count = 1
        checker = 0
        self.trackers = []
        self.paused = paused
        self.cap = cv2.VideoCapture('Tablee.mp4')
        lengthvid = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fgbg = cv2.createBackgroundSubtractorMOG2()
        #Initialisation for VGG
        sess = tf.Session()
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vg.vgg16(images, 'vgg16_weights.npz', sess)
        stat_prob=np.zeros((5,2),dtype=object)
        stat_prob[0,1]=0.1
        stat_prob[1,1]=0.11
        stat_prob[2,1]=0.12
        stat_prob[3,1]=0.13
        stat_prob[4,1]=0.14
        #Initialistaion for Static SURF
        surf=cv2.xfeatures2d.SURF_create(1000,5,5)
        width=self.cap.get(3)
        height=self.cap.get(4)
        frame_overlap=np.zeros((30,3),dtype=float)
        stat_count=0
        fcount = 0
        while(1):
            fcount += 1
            _,self.frame = self.cap.read()
            gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(21,21),0)
            fgmask = fgbg.apply(self.frame)
            thresh = cv2.erode(fgmask, None, iterations=2)
            (_,cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
            count1 = 0
            k = 0
            lstatic = np.array(self.datastatic)
            kstatic = lstatic[1:, 6]
            zstatic = heapq.nlargest(1,kstatic)
            ldynamic = np.array(self.datadynamic)
            kdynamic = ldynamic[1:, 6]
            zdynamic = heapq.nlargest(1,kdynamic)
            key = cv2.waitKey(1) & 0xFF  
            for c in cnts: 
                if cv2.contourArea(c) < 2000 :
                    
                    continue     
                k = k+1
            if cnts != [] or k == 0:

                    
                if count ==1 :

                    for c in cnts:    
                        (x,y,w,h) = cv2.boundingRect(c)
                        self.rect = (x, y, x+w, y+h)                                     
                        if cv2.contourArea(c) < 2000 :
                            continue                                    
                        p = p +1

                        self.onrect(self.rect,count,count1)                    
                        count1 = count1+1
                        checker = run(self,vgg,images,sess,self.frame,cnts,count1,frame_count)
                        count = count +1
                
      
                if checker == 100 or k != p:
                    for dat in self.datadynamic:
                    	f1.write('|'.join(list(map(str,dat))) + '\n')
                    index_static = heapq.nlargest(1,range(len(kstatic)), kstatic.__getitem__)
                    index_dynamic = heapq.nlargest(1,range(len(kdynamic)), kdynamic.__getitem__)        
                    print("{} is moving towards {}".format(self.datadynamic[index_dynamic,5], self.datastatic[index_static,5]))
                    checker = 0
                    
                    self.trackers = []
                    p = 0
                    for c in cnts:    
                        (x,y,w,h) = cv2.boundingRect(c)
                        self.rect = (x, y,x+w, y+h)                                     
                        if cv2.contourArea(c) < 2000 :
                            continue                                    
                        p = p +1

                        self.onrect(self.rect,count,count1)                    
                        count1 = count1+1
                        checker = run(self,vgg,images,sess,self.frame,cnts,count1,frame_count)
                        count = 2            
                else:
                    
                    checker = run(self,vgg,images,sess,self.frame,cnts,count1,frame_count)
                    count1 = count1 +1
                
            else:
                count1 = count1+1
                (kp,descs)=surf.detectAndCompute(gray, None)
                length=len(kp)
                frame_list=np.zeros((length,1),dtype=float)
                for i in range (0,length):
                    frame_list[i,:]=kp[i].response

                index=random.randint(0,length-1)
                xs=int(kp[index].pt[0])
                ys=int(kp[index].pt[1])
                val=int((kp[index].size)/2) 
                new_val=val
                #Resize Attention Window
                if val<50:
                    if xs<100 or ys<100 or xs>(width-100) or ys>(height-100):
                        new_val=val 
                    else:
                        new_val=100
                attention_window = self.frame[ys-new_val:ys+new_val , xs-new_val:xs+new_val]

                if len(self.trackers) > 0:   
                    cv2.imshow('tracker state', self.trackers[-1].state_vis)               
                x0,y0,x1,y1 = self.rect
                #cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)   
                cv2.rectangle(self.frame,(xs-new_val,ys-new_val),(xs+new_val,ys+new_val),(255,0,0),3)
                vgg1 = vg.imresize(attention_window, (224, 224))
                image_stack  = np.stack([vgg1])
                probs = sess.run(vgg.probs, feed_dict={vgg.imgs: image_stack})
                preds = np.argmax(probs, axis=1)
                for index, p in enumerate(preds):
                    cv2.putText(self.frame,"{}".format(class_names[p]),(xs-new_val,ys-new_val),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

                #print("Still,Frame Number:{},Frame Number:{},X-Axis:{}, Y-Axis:{}, Static Prediction:{}, Probablity:{}".format(frame_count,frame_count,xs,ys,class_names[p],probs[index,p]))
                self.datastatic=np.vstack((self.datastatic,np.array(("Still",frame_count,frame_count,xs,ys,class_names[p],probs[index,p]))))
                
            frame_count += 1

            if key == ord("q"):


                break


    def onrect(self, rect, count,count1):
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            tracker = MOSSE(frame_gray, rect,count,count1)            
            self.trackers.append(tracker)


if __name__ == '__main__':
    e1 = cv2.getTickCount()
    print (__doc__)
    import sys, getopt
    global spamwriter
    global save
    f=open("datastatic.csv","w")
    f1=open("datadynamic.csv","w")
    f2=open("beststatic.csv","w")
    checker = 0
    save = []
    opts, args = getopt.getopt(sys.argv[1:], '', ['pause'])
    opts = dict(opts)
    try:
        video_src = args[0]
    except:
        video_src = '0'
    App(video_src, paused = '--pause' in opts)
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print(time)
    print(frame_count)
    print('The time taken to process each frame is',time/frame_count)
