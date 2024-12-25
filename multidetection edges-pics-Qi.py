import cv2
import numpy as np

def nothing(x):
    pass

# 读取TIFF图像
img = cv2.imread('D:/GAIN-LQ-personal/VSCODE/1.tiff')

 # 调整亮度和对比度
img_adjusted = cv2.convertScaleAbs(img, alpha=62/50, beta=0-50)

    # 转换为灰度图像
img_gray = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2GRAY)

    # 去噪
img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 50, 7, 21)

    # 应用阈值
_, img_thresh = cv2.threshold(img_denoised, 112, 255, cv2.THRESH_BINARY)

    # 应用Canny边缘检测
img_canny = cv2.Canny(img_thresh, 54, 150)

cv2.imshow('Adjustments', img_canny)

cv2.waitKey(0)

cv2.destroyAllWindows()

#save the image
cv2.imwrite('D:/GAIN-LQ-personal/VSCODE/1-canny.tiff', img_canny)

#invert the image

img_canny = cv2.bitwise_not(img_canny)
cv2.imwrite('D:/GAIN-LQ-personal/VSCODE/1-canny-2.jpg', img_canny)