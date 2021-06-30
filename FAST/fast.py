import cv2
img = cv2.imread('eiffel.jpg');

## Chuyển lại màu ảnh
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Khởi tạo FAST
fast = cv2.FastFeatureDetector_create(threshold=30)

## Detect các keypoints trong ảnh
keypoints = fast.detect(gray, None)

## Hiển thị các keypoints
result = cv2.drawKeypoints(img, keypoints, None)

stack = cv2.hconcat((img, result))
cv2.imwrite('result.jpg', stack)
cv2.imshow('', stack)

## Hiển thị thông số của thuật toán FAST
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(keypoints)) )

## Tắt nonmaxSupperession
fast.setNonmaxSuppression(0)
keypoints2 = fast.detect(gray, None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(keypoints2)) )
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
