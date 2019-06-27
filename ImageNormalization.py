import numpy as np
import cv2
import os, glob
# read

for img1 in glob.glob("ImageSegment/*.png"):
    img = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 270
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
    cv2.imwrite(img1, imgMorph)
