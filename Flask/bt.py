import cv2 as cv
import numpy as np

#read image
filename = 'objets3.jpg'
img = cv.imread(filename)
cv.imshow('img', img)

#convert to gray image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

#blur image
blur = cv.blur(gray,(3,3))
cv.imshow('blur', blur)


edge = cv.Canny(blur, 50, 300, 3)
cv.imshow('edge', edge)

kernel = np.ones((22, 22), np.uint8)
closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)
cv.imshow('closing', closing)

num_labels, labels_im = cv.connectedComponents(closing)

_, contours, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# for c in contours:
#     # get the bounding rect
#     x, y, w, h = cv.boundingRect(c)
#     # draw a green rectangle to visualize the bounding rect
#     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv.drawContours(img, contours, -1, (0, 255, 0), 2) # vẽ lại ảnh contour vào ảnh gốc
number = 'Number of object: ' + str(num_labels-1)
cv.putText(img, number, (5,25), cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv.imshow('result', img)


if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
