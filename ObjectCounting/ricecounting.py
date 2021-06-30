import cv2 as cv
import numpy as np

#read image
filename = '1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png'
source = cv.imread(filename)
cv.imshow('img', source)

img =cv.cvtColor(source,cv.COLOR_BGR2GRAY)

# Salt Filter
img = cv.medianBlur(img, 7)

# Construct the template, 5 times of corrosion, 5 times of expansion, get the background
kernel=np.ones((5,5),np.uint8)
erosion=cv.erode(img,kernel,iterations=5)
dilation=cv.dilate(erosion,kernel,iterations=5)

#Original image minus the background to get the shape of rice grains
backImg=dilation
rice=img-backImg

#Threshold
th1,ret1=cv.threshold(rice,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#Contour detection
ret1,contours,hierarchy=cv.findContours(ret1,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)


num_labels, labels_im = cv.connectedComponents(ret1)

_, contours, _ = cv.findContours(ret1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(source, contours, -1, (0, 255, 0), 2) # vẽ lại ảnh contour vào ảnh gốc
# number = 'Number of object: ' + str(num_labels-1)
# cv.putText(source, number, (5,25), cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

print('Number of object: ', num_labels-1)
cv.imshow('image',source)
cv.imwrite('salt.jpg', source)
cv.waitKey(0)
cv.destroyAllWindows()
