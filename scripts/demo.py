import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/home/alex/Dropbox/Python/OpenCV")

img = cv2.imread('images/mug.jpg',cv2.IMREAD_GRAYSCALE)
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
