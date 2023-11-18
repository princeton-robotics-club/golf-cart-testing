import numpy as np
import cv2  

# Define points for perspective transform
frame = cv2.imread('square_table.jpg')
print(frame.shape)

pts_src = np.array([[565, 425], [231, 865], [1373, 418], [1707, 850]], dtype=np.float32) # TODO Put coordinates from calibration
pts_dst = np.array([[0, 0], [0, 255], [255, 0], [255, 255]], dtype=np.float32)

# Calculate the transformation matrix
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Apply the perspective transformation
width = frame.shape[1]
height = frame.shape[0]
print(width, height)
transformed_frame = cv2.warpPerspective(frame, matrix, (500, 500))

resized_image = cv2.resize(transformed_frame, (int(width/4), int(height/4)), interpolation=cv2.INTER_AREA)

cv2.imshow('Transformed Frame', resized_image)
cv2.waitKey(0)