
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('eiffel.jpg')
## Chuyển lại màu ảnh
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)

keypoints, desctiptors = orb.detectAndCompute(gray, None)
print("Number of keypoints in image: ", len(keypoints))

dst_with_size = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst_without_size = cv2.drawKeypoints(img, keypoints, None)

stack = cv2.hconcat((dst_with_size, dst_without_size))

cv2.imwrite('result.jpg', stack)
cv2.imshow('',stack)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

