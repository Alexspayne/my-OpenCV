import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Video 16 - Haar Cascades, Object detection.
# Video 15 - Background reduction (Really most like motion detection)
# Video 13 - Corner detection
#This is one where I'll be using video where he uses images.

cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG2()

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
os.chdir("/home/alex/opencv-3.1.0/data/haarcascades/")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


while True:
    _, img = cap.read()
    #16
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    #15
    # fgmask = fgbg.apply(img)
    
    # cv2.imshow('fgmask',img)
    # cv2.imshow('frame',fgmask)
     
#    img = cv2.imread('opencv-corner-detection-sample.jpg')
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)

    # corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    # corners = np.int0(corners)

    # for corner in corners:
    #     x,y = corner.ravel()
    #     cv2.circle(img,(x,y),3,255,-1)
    
#    cv2.imshow('Corner',img)
    
    k = cv2.waitKey(5) & 0xFF 
    if k == 27: #escape key
        break

cv2.destroyAllWindows()
cap.release()


# Video 12 - GrabCut Foreground Extraction
# Video 14 - Feature Matching

# os.chdir("/home/alex/Dropbox/Python/OpenCV/images")

# img1 = cv2.imread('opencv-feature-matching-template.jpg',0)
# img2 = cv2.imread('opencv-feature-matching-image.jpg',0)

# orb = cv2.ORB_create()

# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# matches = bf.match(des1,des2)
# matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
# plt.imshow(img3)
# plt.show()

# img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
# mask = np.zeros(img.shape[:2],np.uint8)

# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)

# rect = (161,79,150,150)

# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]

# plt.imshow(img)
# plt.colorbar()
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Video 11 - Template Matching



# I will just use his images I guess.
# os.chdir("/home/alex/Dropbox/Python/OpenCV/images")   

# img_rgb = cv2.imread('opencv-template-matching-python-tutorial.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# template = cv2.imread('opencv-template-for-matching.jpg',0)
# w, h = template.shape[::-1]


# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.9
# loc = np.where( res >= threshold)


# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# cv2.imshow('Detected',img_rgb)


# cv2.waitKey(0)
# cv2.destroyAllWindows() 

# Video 7, 8, 9, 10 - Color Filtering (*using video feed)

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
    #hue sat value

    # laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    # cv2.imshow('laplacian', laplacian)

    # sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    # sobelxy = cv2.Sobel(sobelx, cv2.CV_64F, 0, 1, ksize=5)
    # sobelyx = cv2.Sobel(sobely, cv2.CV_64F, 1, 0, ksize=5)


    # cv2.imshow('sobelxy', sobelxy)
    # cv2.imshow('sobelyx', sobelyx)

    # edges = cv2.Canny(frame, 100, 100)
    # cv2.imshow('edges', edges)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower_red = np.array([150,150,50])
    # upper_red = np.array([180,255,150])

    # # dark_red = np.uint8([[[12,22,121]]])
    # # dark_red = cv2.cvtColor(dark_red,cv2.COLOR_BGR2HSV)   
    
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(frame, frame, mask = mask)


    # kernel = np.ones((5,5), np.uint8)
    # erosion = cv2.erode(mask, kernel, iterations = 1)
    # dilate = cv2.dilate(mask, kernel, iterations = 1)
    # # cv2.imshow('erosion', erosion)
    # # cv2.imshow('dilation', dilate)

    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('opening', opening)
    # cv2.imshow('closing', closing)
    #    smoothed = cv2.filter2D(res,-1,kernel)
    #    cv2.imshow('smoothed', smoothed)    

    #    cv2.imshow('frame', frame)
    #    cv2.imshow('mask', mask)
    #    cv2.imshow('res', res)
    #    blur = cv2.GaussianBlur(res,(15,15),0)
    #    cv2.imshow('blur', blur)

    # median = cv2.medianBlur(res,15)
    # cv2.imshow('median blur', median)

    #    bilateral = cv2.bilateralFilter(res,15,75,75)
    #    cv2.imshow('bilateral filter', bilateral)

#     k = cv2.waitKey(5) & 0xFF 
#     if k == 27: #escape key
#         break

# cv2.destroyAllWindows()
# cap.release()




# Video 6 - Thresholding
# os.chdir("/home/alex/Dropbox/Python/OpenCV")

# img = cv2.imread('images/bookpage.jpg')

# retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

# grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)

# gauss = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                   cv2.THRESH_BINARY, 115, 1)

# retval2, otsu = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_OTSU)

# cv2.imshow('original', img)
# # cv2.imshow('threshold', threshold)
# # cv2.imshow('threshold', threshold2)       
# # cv2.imshow('threshold', gauss) #Best for this
# cv2.imshow('threshold', otsu)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Bonus:
#I'm doing this with video for more fun.
#Wait... maybe I shouldn't?
# No, I should. Nah. I'll do it if he does it.
# If he doesn't do it for a while, then I'll do it.

# Video 5 and previous code below:

# img1 = cv2.imread('images/Dice.png')
# #img2 = cv2.imread('images/Black_Stapler.jpg')
# imgsmall = cv2.imread('images/mug.jpg')


# rows,cols,channels = imgsmall.shape
# roi = img1[0:rows,0:cols]

# img2gray = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

# #cv2.imshow('mask',mask)

# mask_inv = cv2.bitwise_not(mask)
# #bitwise is just like and/or in python
# img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# img2_fg = cv2.bitwise_and(imgsmall, imgsmall, mask=mask)

# dst = cv2.add(img1_bg, img2_fg)
# img1[0:rows,0:cols] = dst

# cv2.imshow('res',img1)
# cv2.imshow('mask_inv',mask_inv)
# cv2.imshow('img1_bg',img1_bg)
# cv2.imshow('img2_fg',img2_fg)
# cv2.imshow('dst',dst)

#image = img1 + img2
#image = cv2.add(img1,img2)

#weighted
#image = cv2.addWeighted(img1, .6, img2, .4,0)



# img[55,55] = [200,200,200]
# px = img[55,55]

# #Region of Image
# img[100:150, 100:150] = [200,200,200]

# cup_mouth = img[200:250,200:250]
# img[0:50, 0:50] = cup_mouth

# cv2.line(img, (0,0), (150,150), (255,255,255), 15)

# cv2.rectangle(img, (15,25), (200,150), (0,255,0), 5)

# cv2.circle(img, (100,63), 55, (0,0,255), -1)
# #IMREAD_COLOR = 1
# #IMREAD_UNCHANGED = -1

# #writing
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'Good fun my good buddy', (0,130), font, 2, (200,255,255), 5, cv2.LINE_AA)

# cv2.imshow('image',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
