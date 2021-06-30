import cv2
import numpy as np
img = cv2.imread('eiffel.jpg');
## Chuyển ảnh về ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Khởi tạo SIFT và tính toán keypoints, descriptors
surf = cv2.xfeatures2d.SURF_create()
keypoints, descriptors = surf.detectAndCompute(gray, None)
print("Number of keypoints in image:", len(keypoints))

## Vẽ các keypoints và hiển thị lên ảnh
## Hàm drawKeypoints được cung cấp bởi OpenCV để vẽ các keypoint lên ảnh.
## Input: img(1): Ảnh đầu vào, keypoints: Danh sách các keypoints, img(2): Ảnh đầu ra
## flag: cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS để hiện thị thêm kích thước và hướng của keypoint
cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

## Tính toán matrix xoay ảnh
rows, cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D(center=(cols/2, rows/2), angle=-40, scale=0.5)

## Xoay ảnh theo matrix vừa tính
img1 = cv2.warpAffine(img, rotation_matrix, (cols, rows))

## Ghép 2 ảnh và hiển thị
stack = cv2.hconcat((img, img1))
cv2.imshow('',stack)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

img_keypoints, img_dcts = sift.detectAndCompute(gray, None)
print("Number of keypoints in img: ", len(img_keypoints))

img1_keypoints, img1_dcts = sift.detectAndCompute(gray1, None)
print("Number of keypoints in img1: ", len(img1_keypoints))

img_clone = np.copy(img)
img1_clone = np.copy(img1)

img_draw_keypoints = cv2.drawKeypoints(img, img_keypoints, img_clone, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_draw_keypoints = cv2.drawKeypoints(img1, img1_keypoints, img1_clone, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

stack = cv2.hconcat((img_draw_keypoints, img1_draw_keypoints))

cv2.imshow('',stack)

## Khởi tạo Brute Force Matcher
## Input: normType: loại hàm tính toán khoảng cách. Đối với SIFT, SURF thì sử dụng cv2.NORM_L1, cv2.NORM_L2. Đối với ORB, BRIEF, BRISK thì sử dụng cv2.NORM_HAMMING
## crossCheck: default False, nếu True thì 2 desciptors ở 2 ảnh phải khớp nhau mới match với nhau.
bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

## Tính toán các điểm có khả năng match với nhau.
matches = bf.match(img_dcts, img1_dcts)

## Sort lại các desciptors match nhau theo thứ tự khoảng cách lớn dần.
matches = sorted(matches, key = lambda x:x.distance)

## Tạo 1 mảng lưu kết quả và vẽ 30 matches đầu tiên trên ảnh đó
result = np.copy(stack)
cv2.drawMatches(img, img_keypoints,img1, img1_keypoints, matches[:30], result, flags=2)

cv2.imwrite('matching.jpg', result)
cv2.imshow('',result)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
