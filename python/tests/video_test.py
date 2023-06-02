import cv2
import numpy as np
 

frameSize = (500, 500)

out = cv2.VideoWriter('output_video_1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

for i in range(0,255):
    img = np.ones((500, 500, 3), dtype=np.uint8)*i
    out.write(img)

out.release()