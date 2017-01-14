import cv2
import numpy as np
#import matplotlib.pyplot as plt
import os

os.chdir("/home/alex/Dropbox/Python/OpenCV")

img = cv2.imread('images/mug.jpg',cv2.IMREAD_COLOR)

img[55,55] = [200,200,200]
px = img[55,55]

#Region of Image
img[100:150, 100:150] = [200,200,200]

cup_mouth = img[200:250,200:250]
img[0:50, 0:50] = cup_mouth

# cv2.line(img, (0,0), (150,150), (255,255,255), 15)

# cv2.rectangle(img, (15,25), (200,150), (0,255,0), 5)

# cv2.circle(img, (100,63), 55, (0,0,255), -1)
# #IMREAD_COLOR = 1
# #IMREAD_UNCHANGED = -1

# #writing
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'Good fun my good buddy', (0,130), font, 2, (200,255,255), 5, cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
