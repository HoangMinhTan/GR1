import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('1_wIXlvBeAFtNVgJd49VObgQ_sinus.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(np.abs(fshift), cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

fshift[230,222]=0
fshift[230,238]=0

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(np.abs(fshift), cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

img_back = img_back.astype('uint8')

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

cv.drawContours(img, contours, -1, (0, 255, 0), 2) # vẽ lại ảnh contour vào ảnh gốc

print('Number of object: ', num_labels-1)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
