import cv2
import numpy as np


def nothing(x):
    pass

# 读取TIFF图像
img = cv2.imread('D:/opencv/opencv/project-qi-241224/1.tiff')
if img is None:
    print("Error: Could not open or find the image.")
    exit()

height, width = img.shape[:2]
# 创建窗口
cv2.namedWindow('Adjustments')
cv2.resizeWindow('Adjustments', width, height)

# 创建trackbars
cv2.createTrackbar('Brightness', 'Adjustments', 50, 100, nothing)
cv2.createTrackbar('Contrast', 'Adjustments', 50, 100, nothing)
cv2.createTrackbar('Canny Min', 'Adjustments', 50, 255, nothing)
cv2.createTrackbar('Canny Max', 'Adjustments', 150, 255, nothing)
cv2.createTrackbar('Threshold', 'Adjustments', 127, 255, nothing)
cv2.createTrackbar('Denoise', 'Adjustments', 10, 50, nothing)

while True:
    # 获取trackbar位置
    brightness = cv2.getTrackbarPos('Brightness', 'Adjustments') - 50
    contrast = cv2.getTrackbarPos('Contrast', 'Adjustments') / 50.0
    canny_min = cv2.getTrackbarPos('Canny Min', 'Adjustments')
    canny_max = cv2.getTrackbarPos('Canny Max', 'Adjustments')
    threshold_val = cv2.getTrackbarPos('Threshold', 'Adjustments')
    denoise_val = cv2.getTrackbarPos('Denoise', 'Adjustments')

    # 调整亮度和对比度
    img_adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # 转换为灰度图像
    img_gray = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2GRAY)

    # 去噪
    img_denoised = cv2.fastNlMeansDenoising(img_gray, None, denoise_val, 7, 21)

    # 应用阈值
    _, img_thresh = cv2.threshold(img_denoised, threshold_val, 255, cv2.THRESH_BINARY)

    # 应用Canny边缘检测
    img_canny = cv2.Canny(img_thresh, canny_min, canny_max)

    # 显示结果
    cv2.imshow('Adjustments', img_canny)

    # 按下ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

img=cv2.imread('D:/opencv/opencv/project-qi-241224/1.tiff')
img=cv2.convertScaleAbs(img,alpha=50,beta=70)
img=cv2.threshold(img,153,255,cv2.THRESH_BINARY)
img=cv2.fastNlMeansDenoising(img,None,30,7,21)
img1=cv2.canny(img,50,102)

cv2.imshow('img',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('D:/opencv/opencv/project-qi-241224/1-1.tiff',img1)
