import cv2

img = cv2.imread('eiffel.jpg');
## Chuyển ảnh về ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Khởi tạo SIFT và tính toán keypoints, descriptors
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
print("Number of keypoints in image:", len(keypoints))

## Vẽ các keypoints và hiển thị lên ảnh
## Hàm drawKeypoints được cung cấp bởi OpenCV để vẽ các keypoint lên ảnh.
## Input: img(1): Ảnh đầu vào, keypoints: Danh sách các keypoints, img(2): Ảnh đầu ra
## flag: cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS để hiện thị thêm kích thước và hướng của keypoint
cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('result.jpg',img)

cv2.imshow('result',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

