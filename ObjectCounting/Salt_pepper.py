import cv2 as cv
import numpy as np

#read image
filename = '1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png'
img = cv.imread(filename)
cv.imshow('img', img)

#median filter
median = cv.medianBlur(img, 5)
cv.imshow('median', median)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
