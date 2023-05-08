import cv2
import sys
import numpy as np

# Parameter
N = 46 # number of properties
fw = 2048 # frame width
fh = 1536 # frame height
setFile = 'settings.txt' # file name with settings

# Open the device at the ID 0
print('Try to open camera...')
cap = cv2.VideoCapture(0)

# Check whether camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")
    sys.exit(0)

print('Camera opened successfully')

# Get Camera setting
print('Read settings...')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
props = [str(cap.get(i)) for i in range(N)]

# Write data to file
print('Write settings to "{}"...'.format(setFile))

with open(setFile, "w+") as f:
    f.write(','.join(props))
    
cap.release()

cv2.destroyAllWindows()
print('Settings successfully written to "{}"'.format(setFile))