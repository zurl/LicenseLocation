import numpy as np
from functools import partial
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt


def detect_licese(file, norm, enhance=True, kill_curve=False, showImg=True, enhance_speed=2, vertical_coe = 12, itercoe=4, hwRate=(1.8, 8.8),
                  height_range=(0.4, 0.9), square_range=(0.001, 0.5),
                  vertical_range=(0.1, 0.9),range_coe = 1.1,anit_verical_coe=6):
    def show(img):
        cv2.imshow(file,img)
        cv2.waitKey(0)
    ifile = cv2.imread(file)
    oriX = len(ifile)
    oriY = len(ifile[0])
    # img0 = cv2.drawContours(ifile,np.array([[[int(norm[2*j]),int(norm[2*j+1])] for j in range(0,4)]]),-1,(255,255,0))
    # show(img0[::3,::3])

    img = cv2.resize(ifile,(384,288))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Image Enhancement
    base_vertex = [(x,y) for x in range(0,9) for y in range(0,9)]
    pos_vertex = map(lambda x:(x[0]*36,x[1]*48),base_vertex)
    vermap = lambda x,y:x*8+y
    IWmean = [ img[
        0   if i[0] == 0 else i[0] - 18:
        288 if i[0] == 288 else i[0] + 18,
        0   if i[1] == 0 else i[1] - 24:
        384 if i[1] == 384 else i[1] + 24]
                 .mean() for i in pos_vertex]
    IWstd = [ img[
        0   if i[0] == 0 else i[0] - 18:
        288 if i[0] == 288 else i[0] + 18,
        0   if i[1] == 0 else i[1] - 24:
        384 if i[1] == 384 else i[1] + 24]
                 .std() for i in pos_vertex]
    # Cx = (j-48n) * 1.0/48
    # Cy = (i-36m) * 1.0/36
    # Imean(i,j) = (1-Cy) * ((1 - Cx) * IWmean(A) + CxIWmean(B))
    #              + Cy * ((1 - Cx) * IWmean(C) + CxIWmean(D))
    # Istd(i,j)  = (1-Cy) * ((1 - Cx) * IWstd(A) + CxIWstd(B))
    #              + Cy * ((1 - Cx) * IWstd(C) + CxIWstd(D))
    def getImean(i,j):
        m = i//36
        n = j//48
        Cx = (j-48*n) * 1.0/48
        Cy = (i-36*m) * 1.0/36
        Imean = (1-Cy) * ((1 - Cx) * IWmean[vermap(m,n)]   + Cx*IWmean[vermap(m,n+1)])\
                  + Cy * ((1 - Cx) * IWmean[vermap(m+1,n)] + Cx*IWmean[vermap(m+1,n+1)])
        Istd  = (1-Cy) * ((1 - Cx) * IWstd[vermap(m,n)]    + Cx*IWstd[vermap(m,n+1)])\
                  + Cy * ((1 - Cx) * IWstd[vermap(m+1,n)]  + Cx*IWstd[vermap(m+1,n+1)])
        return Imean,Istd
    def getCoe(std):
        if std < 20:
            return (enhance_speed + 1.0) / ((enhance_speed * 1.0/ 400) * (std - 20) *(std - 20) + 1)
        elif std < 60:
            return (enhance_speed + 1.0)  / ((enhance_speed * 1.0/ 1600) * (std - 20) *(std - 20) + 1)
        else:
            return 1
    def doImap(img):
        ret = np.empty_like(img)
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                val = getImean(i,j)
                coe = getCoe(val[1])
                ret[i,j] = coe * (img[i,j] - val[0]) + val[0]
        return ret
    img2 = doImap(img)
    img2 = cv2.convertScaleAbs(cv2.Sobel(img2 if enhance else img,cv2.CV_64F,2,0,7))
    tmp,img3 = cv2.threshold(img2,75,255,cv2.THRESH_BINARY)
    # Curve remove
    if kill_curve:
        M = np.zeros_like(img)
        N = np.zeros_like(img)
        Tlong = 160
        Tshort = 50
        for i in range(2,286):
            for j in range(2,382):
                if img3[i, j] == 255:
                    if img3[i-1, j-1] + img3[i-1, j] + img3[i-1, j+1] + img3[i, j-1] > 0:
                        M[i, j] = max(M[i-1, j-1], M[i-1, j], M[i-1, j+1], M[i, j-1]) + 1
                    else:
                        M[i, j] = max(M[i-2, j-1], M[i-2, j], M[i-2, j+1],
                                     M[i-1, j-2], M[i-1, j+2], M[i, j-2]) + 1
        for i in range(2,286)[::-1]:
            for j in range(2,382)[::-1]:
                if img3[i, j] == 255:
                    if img3[i+1,j-1] + img3[i+1, j] + img3[i+1, j+1] + img3[i, j+1] > 0:
                        N[i, j] = max(N[i+1, j-1], N[i+1, j], N[i+1, j+1], N[i, j+1]) + 1
                    else:
                        N[i, j] = max(N[i+2, j-1], N[i+2, j], N[i+2, j+1],
                                     N[i+1, j-2], N[i+1, j+2], N[i, j+2]) + 1
        for i in range(0,288):
            for j in range(0,384):
                if img3[i,j] == 255:
                    if M[i,j] + N[i,j] >Tlong or M[i,j] + N[i,j] < Tshort:
                        img3[i,j] = 0


    def cluster(img):
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (vertical_coe, 1))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelX)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernelX)
    def antiNoise(img):
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT,(1, anit_verical_coe))
        closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelX)
        return cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernelX)

    img4 = cluster(antiNoise(img3))
    img5 = cv2.erode(img4, None, iterations = itercoe)
    img5 = cv2.dilate(img5, None, iterations = itercoe)

    contours, hierarchy,_ = cv2.findContours(img5.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # rect find
    img0 =  cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2RGB)
    rect = map(cv2.minAreaRect,hierarchy)
    rect = map(cv2.boxPoints,rect)
    cv2.drawContours(img0,map(lambda x:x.astype(np.int),rect),-1,(0,255,0))
    rect = map(cv2.boundingRect,rect)
    # nearby rect merge
    # !!!!Work not well
    # for x in range(0,len(rect)):
    #     for y in range(0,len(rect)):
    #         if x != y and \
    #             (rect[y][0] - rect[x][0] - rect[x][2] <= 384 * mergecoe):
    #             rect[x] = (rect[x][0],rect[x][1],rect[y][0] - rect[x][0] + rect[y][0]
    #                         ,max(rect[x][3],rect[y][1]+rect[y][3]-rect[x][1]))
    #             rect[y] = (0,0,0,0)
    #rect filter
    def squareFilter(x):
        area = x[2] * x[3]
        return area <= 288 * 384 * square_range[1] and area >= 288 * 384 * square_range[0]
    def heightFilter(x):
        return x[1] <= 288 * height_range[1] and x[1] >= 288 * height_range[0]\
               and (x[1] + x[3] <= 288 * height_range[1]
                    or ( x[0] >=  384 * 0.3 and x[0] <= 384 * 0.6))
    def hwRateFilter(x):
        return x[2] > x[3] * hwRate[0] and x[2] < x[3] * hwRate[1]
    def vertialFilter(x):
        return x[0] >= vertical_range[0] * 384 and x[0] + x[2] <= vertical_range[1] * 384
    def draw_rect(x,color):
        cv2.rectangle(img0,(x[0],x[1]),(x[0]+x[2],x[1]+x[3]),color)
    rect = filter(squareFilter,rect)
    rect = filter(heightFilter,rect)
    rect = filter(hwRateFilter,rect)
    rect = filter(vertialFilter,rect)
    def posChangeRange(x):
        dx = int(x[2] * (range_coe - 1))
        dy = int(x[3] * (range_coe - 1))
        return x[0]-dx,x[1]-dy,x[2]+2*dx,x[3] + 2*dy
    rect = map(posChangeRange,rect)
    # rect draw
    map(partial(draw_rect,color=(255,0,0)),rect)
    # rect selection
    Good = (384 * 0.5,288 * 0.75)

    def rectCmp(x, y):
        centerX = (x[0]+x[2]*1.0/2,x[1] + x[3]*1.0/2)
        centerY = (y[0]+y[2]*1.0/2,y[1] + y[3]*1.0/2)
        return (centerX[0] - Good[0]) ** 2 + (centerX[1] - Good[1]) ** 2 >\
            (centerY[0] - Good[0]) ** 2 + (centerY[1] - Good[1]) ** 2
    def posChange(x):
        return x[0]*1.0/384*oriY, x[1]*1.0/288*oriX, x[2]*1.0/384*oriY, x[3]*1.0/288*oriX
    img0 = cv2.drawContours(img0,np.array([[[int(norm[2*j]*1.0/oriY*384),int(norm[2*j+1]*1.0/oriX*288)] for j in range(0,4)]]),-1,(255,255,0))
    rect.sort(rectCmp)
    if(showImg):
        show(img0)
    return map(posChange,rect)
