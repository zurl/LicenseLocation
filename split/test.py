import split as sp
import numpy as np
import cv2
def show(img,strs):
    cv2.imshow(strs,img)
    cv2.waitKey(0)
label_file = open('char.txt')
label = label_file.readlines()
no = map(lambda x:x[0:4],label)
la = map(lambda x:x[6:9],label)
lb = map(lambda x:x[9:15],label)
print la
print lb
for _ in range(0,100):
    try :
        ret,img = sp.char_split(no[_])
        ret = ret[::-1]
        for __ in range(0,7):
            y = ret[__]
            tmp = img[y[1]:y[1]+y[3],y[0]:y[0]+y[2]]
            if __ == 0:
                print la[_][::-1]
                show(tmp,str(_*100+__))
            else:
                print lb[_][__-1]
                show(tmp,str(_*100+__))

    except:
        pass
print label
print no
