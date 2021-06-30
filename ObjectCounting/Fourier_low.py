import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
source = cv.imread('1_zd6ypc20QAIFMzrbCmJRMg.png',0)

img = np.power(source, 0.1)
max_val = np.max(img.ravel())
img = img / max_val * 255
img_back = img.astype(np.uint8)

cv.imshow('source', source)

# Construct the template, 5 times of corrosion, 5 times of expansion, get the background
kernel=np.ones((5,5),np.uint8)
erosion=cv.erode(img_back,kernel,iterations=5)
dilation=cv.dilate(erosion,kernel,iterations=5)

#Original image minus the background to get the shape of rice grains
backImg=dilation
rice=img_back-backImg

#OSTU Binarization
th1,ret1=cv.threshold(rice,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#Contour detection
ret1,contours,hierarchy=cv.findContours(ret1,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

num_labels, labels_im = cv.connectedComponents(ret1)

_, contours, _ = cv.findContours(ret1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img = cv.cvtColor(img_back, cv.COLOR_GRAY2BGR)
cv.drawContours(img, contours, -1, (0, 255, 0), 2) # vẽ lại ảnh contour vào ảnh gốc
# number = 'Number of object: ' + str(num_labels-1)
# cv.putText(img, number, (5,25), cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

print('Number of object: ', num_labels-1)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
