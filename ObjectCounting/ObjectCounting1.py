import cv2 as cv
import numpy as np

#read image
filename = 'objets1.jpg'
img = cv.imread(filename)
cv.imshow('img', img)

blur = cv.blur(img, (5,5))

#convert to lab color
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

edge = cv.Canny(lab,10, 50)
cv.imshow('edge', edge)


kernel = np.ones((20, 20), np.uint8)
closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)
cv.imshow('closing', closing)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
cv.imshow('opening', opening)

num_labels, labels_im = cv.connectedComponents(opening)

_, contours, _ = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0, 255, 0), 2) # vẽ lại ảnh contour vào ảnh gốc
number = 'Number of object: ' + str(num_labels-1)
cv.putText(img, number, (5,25), cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

cv.imshow('result', img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
