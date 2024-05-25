import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
base_dir = "C:/Users/AyonA333/Desktop/"
path_test_image = os.path.join(base_dir, "data.png")
image_color = cv2.imread(path_test_image)
new_shape = (image_color.shape[1], image_color.shape[0])
image_color = cv2.resize(image_color, new_shape)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY_INV, 11, 2)
 
cv2.imshow('binary image', adaptive_threshold)
cv2.waitKey(0)