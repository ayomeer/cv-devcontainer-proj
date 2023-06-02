import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sift_visualizer as sv
## 1. Read images
# Read train image
#trainImage = cv.imread('praktika/exercises/local-features/168.png')
trainImage = cv.imread('praktika/exercises/local-features/stapleRemover.jpg')
plt.figure()
plt.imshow(trainImage, cmap='gray')
plt.title('Image of a Box')
plt.show()

# Read query image
#queryImage = cv.imread('praktika/exercises/local-features/CBL8S_on_stack.png')
queryImage = cv.imread('praktika/exercises/local-features/clutteredDesk.jpg')
plt.figure()
plt.imshow(queryImage, cmap='gray')
plt.title('Query Image with a Cluttered Scene')
plt.show()

## 2. Detect keypoints, compute descriptors and display them
# Detect Keypoints
sift = cv.SIFT_create(
    nfeatures=3000,
    contrastThreshold=0.001,
    edgeThreshold=20,
    sigma=1.5,
    nOctaveLayers=4
)
trainPoints = sift.detect(trainImage, None)
queryPoints = sift.detect(queryImage, None)
# Compute descriptors
_, trainDescriptors = sift.compute(trainImage, trainPoints)
_, queryDescriptors = sift.compute(queryImage, queryPoints)

# Visualize keypoints
siftVisualizer = sv.SiftVisualizer(trainImage, trainPoints, trainDescriptors)
siftVisualizer.investigator(imageName='Train Image: Right-click to investigate descriptor')

siftVisualizer = sv.SiftVisualizer(queryImage, queryPoints, queryDescriptors)
siftVisualizer.investigator(imageName='Query Image: Right-click to investigate descriptor')

# 3. Match keypoints and show the matches
# Brute force matching
matcher = cv.BFMatcher_create(cv.NORM_L1, crossCheck=True)
matches = matcher.match(queryDescriptors, trainDescriptors)
print('{} matches found'.format(len(matches)))

# Show matches, starting with the most reliable
sortedMatches = sorted(matches, key = lambda x:x.distance)
pltImage = cv.drawMatches(queryImage, queryPoints, trainImage, trainPoints, sortedMatches[:400], queryImage, flags=2)
plt.imshow(pltImage)
plt.title('Brute force matching result')
plt.show()

# 4. Fit the homography model into the found keypoint correspondences robustly and get a mask of inlier matches:
dstPtsCoords = np.float32([queryPoints[m.queryIdx].pt for m in matches]).reshape(-1,2)
srcPtsCoords = np.float32([trainPoints[m.trainIdx].pt for m in matches]).reshape(-1,2)
H, mask = cv.findHomography(srcPoints=srcPtsCoords, dstPoints=dstPtsCoords, method=cv.RANSAC, ransacReprojThreshold=4.0)
# H, mask = cv.estimateAffinePartial2D(from_=srcPtsCoords, to=dstPtsCoords, inliers=None, method=cv.RANSAC, ransacReprojThreshold=25.0)

matchesMask = mask.ravel().tolist()

# 5. Draw matches
imgMatches = cv.drawMatches(
    img1=queryImage,
    keypoints1=queryPoints,
    img2=trainImage,
    keypoints2=trainPoints,
    matches1to2=matches,
    outImg=None,
    matchesThickness=1,
    matchColor = None,  # (100,255,100), # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask,  # draw only inliers
    flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,  # cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
plt.imshow(imgMatches)
plt.title('Matches refined with homography-RANSAC')
plt.show()
