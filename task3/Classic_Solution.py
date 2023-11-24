import numpy as np
import cv2

File_Name = '2.png'
Image = np.array(cv2.imread(File_Name, cv2.IMREAD_GRAYSCALE))

for i in range(Image.shape[0]):
    for j in range(Image.shape[1]):
        if Image[i][j] > 128:
            Image[i][j] = 255
        if Image[i][j] < 128:
            Image[i][j] = 0

cv2.imshow("M16_noise", Image)
cv2.waitKey(0)

