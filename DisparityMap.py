import numpy as np
import cv2
from matplotlib import pyplot as plt

print("Started Disparity Map Generator Program....")

plt.subplot(3, 1, 1)
print("Reading Left Image...")
imgL = cv2.imread('DisparityL.jpg',0)
plt.imshow(imgL,'gray')

plt.subplot(3, 1, 2)
print("Reading Right Image...")
imgR = cv2.imread('DisparityR.jpg',0)
plt.imshow(imgR,'gray')

print("Finished Reading Images.")

print("Computing Disparity Map...")

stereo = cv2.StereoBM_create(numDisparities=0, blockSize=21)
disparity = stereo.compute(imgR,imgL)

print("Showing Disparity Map...")
plt.subplot(3, 1, 3)
plt.imshow(disparity,'gray')
plt.show()