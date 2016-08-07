import detect as dt
import numpy as np
import cv2
label_file = open('detectlabel.txt')
label = label_file.readlines()[350:400]
no = map(lambda x:x[0:4],label)
no = no
label1 =[x[0:len(x)-1].split('*') for x in label]
location = map(lambda x:map(lambda x:map(lambda x:int(x),filter(lambda x:x!='',x.split(' '))),x[1:]),label1)
def judgeInclude(i,box):
    for j in range(0,4):
        if not( box[0] <= i[2*j] <= box[0] + box[2]and box[1] <= i[2*j + 1] <= box[1] + box[3]):
            return False
    return True
def judgeSize(i,box):
    s1 = cv2.contourArea(np.array([[i[2*j],i[2*j+1]] for j in range(0,4)]))
    s2 = box[2] * box[3]
    return s1 * 5.8 >= s2
def judgeCorrectness(i,box):
    return judgeInclude(i,box) and judgeSize(i,box)

corr1 = 0
corr3 = 0
err1 = 0
err3 = 0
for _ in range(0,len(no)):
    x = no[_]
    ret = dt.detect_licese("pic\\" + x + ".jpg", location[int(_)][0],
                           vertical_coe=14, square_range=(0.007, 0.27), enhance=True,
                           hw_rate=(1.1, 11), height_range=(0.20, 0.91), enhance_speed=2.2,
                           iter_coe=6, anit_verical_coe=6, range_coe = 1.15, show_img=False, kill_curve=False)
    if len(ret) == 0:
        err3 += 1
        err1 += 1
        continue
    if judgeCorrectness(location[int(_)][0],ret[0]):
        corr1 += 1
    else:
        err1 += 1
    yes3 = False
    for y in ret[:3]:
        if judgeCorrectness(location[int(_)][0],y):
            yes3 = True
    if yes3:
        corr3 += 1
    else:
        err3 += 1
print("Correct1 : %f" % (corr1 * 1.0 / (corr1+err1)))
print("Correct3 : %f" % (corr3 * 1.0 / (corr3+err3)))

