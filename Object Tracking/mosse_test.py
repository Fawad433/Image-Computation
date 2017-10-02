# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:53:06 2017

@author: Yashad
"""

from __future__ import print_function

import sys

PY3 = sys.version_info[0] == 3



if PY3:

    xrange = range



import numpy as np

import cv2
import imutils
import os
import common
import csv
#import video



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
        #self.spamwriter = spamwriter
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



    def draw_state(self, vis):

        (x, y), (w, h) = self.pos, self.size

        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))

        if self.good:

            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

        else:

            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))

        #common.draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)
        text = str(self.count1) 
        cv2.putText(vis,"{}".format(text),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        save.append(np.hstack((np.hstack((x1,y1)),np.hstack((x2,y2)),np.hstack((self.count,self.count1)))))
#        aa = zip([x1],[y1],[x1+x2],[y1+y2],[self.count],[self.count1])
#        spamwriter.writerow(aa)
        
        



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
       
        def run(self):
            #print(3)
            while True:
                
                if not self.paused:
                    
                    ret, self.frame = self.cap.read()
    
                    if not ret:
    
                        break
    
                    frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    
                    for tracker in self.trackers:
    
                        tracker.update(frame_gray)
    
    
    
                vis = self.frame.copy()
    
                for tracker in self.trackers:
    
                    tracker.draw_state(vis)
    
                if len(self.trackers) > 0:
    
                    cv2.imshow('tracker state', self.trackers[-1].state_vis)
                
                x0,y0,x1,y1 = self.rect
    
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    
                print(1)
    
                cv2.imshow('frame', vis)
    
                ch = cv2.waitKey(10)
    
                #if ch == 27:
    
                break
                
                if ch == ord(' '):
                    
                    self.paused = not self.paused
    
                if ch == ord('c'):
    
                    self.trackers = []
                    
        p = 0
        k = 0
        e1 = cv2.getTickCount()
        cnts1 = 0
        count = 0
        cnts = 0 
        save1 = []
        save2 = []
        save3 = []
        save4 = []
        save5 = []
        save = []
        self.count = count
        self.cnts = cnts
        count1 = 0
        self.trackers = []
        self.paused = paused
        firstFrame = 0
        self.cap = cv2.VideoCapture('walking.avi')
        with open('eggs.csv', 'w') as c:
            spamwriter = csv.writer(c, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            aa = zip('x','y','w','h','frame','track0','x1','y1','w1','h1','frame','track1','x2','y2','w2','h2','frame','track2','x3','y3','w3','h3','frame','track3','x4','y4','w4','h4','frame','track4')
            spamwriter.writerow(aa)
            while(1):
                
                _,self.frame = self.cap.read()
                gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray,(21,21),0)
                if count < 3:
                    

                    if count==0:
                        firstFrame = gray
                        p = p+1
                        count = count+1        
                    else:
                        #print(0)
                        frameDelta = cv2.absdiff(firstFrame,gray)
                        thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]
                        thresh = cv2.dilate(thresh,None,iterations=2)
                        (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
                        count1 = 0
                        
                        for c in cnts:    
                            
                                
                                (x,y,w,h) = cv2.boundingRect(c)
                                self.rect = (x, y, x+80, y+100) 
                                
                                if cv2.contourArea(c) < 2000 :
                                    continue
                                          
                                self.onrect(self.rect,count,count1)
                                        
                                count1 = count1+1
                                run(self)
                                    
                                count = count+1   
                k = k+1


       
                if count >= 3:
                    count = count +1 
                    print("please")
                    run(self)
                    if count == 20:
                        count = 0
                        self.trackers = []

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    len(save)
                    break
        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        print(k/time)
        for i in range (len(save)):
            if (save[i])[5] == 0:
                save1.append(save[i])
                
            if (save[i])[5] == 1:
                save2.append(save[i])
                
            if (save[i])[5] == 2:
                save3.append(save[i])
                
            if (save[i])[5] == 3:
                save4.append(save[i])
                
            if (save[i])[5] == 4:
                save5.append(save[i])
        for i in range (min(len(save1),len(save2),len(save3),len(save4),len(save5))):
            
            aa = zip([save1[i]],[save2[i]],[save3[i]],[save4[i]],[save5[i]])
            spamwriter.writerow(aa)

                    



    def onrect(self, rect, count,count1):
           # print(2)
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            tracker = MOSSE(frame_gray, rect,count,count1)
            
            self.trackers.append(tracker)

            







if __name__ == '__main__':

    print (__doc__)

    import sys, getopt
    global spamwriter
    global save 
    save = []
    opts, args = getopt.getopt(sys.argv[1:], '', ['pause'])

    opts = dict(opts)

    try:

        video_src = args[0]

    except:

        video_src = '0'



    App(video_src, paused = '--pause' in opts)
        
        
        
        
        
        
        
        
        
        
        