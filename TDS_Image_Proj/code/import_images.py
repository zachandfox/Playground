# Import Packages
import skimage
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

import os
import glob

def import_images(input_directory):
    # Instantiate Search Parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    # Filter by Area.
    params.filterByArea = True
    #params.minArea = 1500
    params.maxArea = 100
    #filter by color
    params.filterByColor = 1


    # check opencv version and construct the detector
        # Ref: https://stackoverflow.com/questions/48136978/how-to-use-feature2dsuch-as-simpleblobdetector-correctly-python-opencv/48137140
    is_v2 = cv2.__version__.startswith("2.")
    if is_v2:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Load Images
    imset = np.array([cv2.imread(i) for i in glob.glob(str(input_directory)+"*.jpg")])

    # Create Output folder
    print(not os.path.exists(directory+"output"))
    if (not os.path.exists(directory+"output")):
      os.mkdir(str(directory)+"output")

    #Iterate process over set of images
    for i in range(len(imset)):
        img = imset[i]
        keypoints = detector.detect(img)

        # Grab points
        points = [[keypoints[i].pt[0], keypoints[i].pt[1]] for i in range(4)]

        # define the input and output image points to warp
        inpts = np.float32(points)
        outpts = np.float32([[0, 0],[600, 0],[0, 600],[600, 600]])

        # calculate a perspective transform based on the points given above
        M = cv2.getPerspectiveTransform(inpts, outpts)

        # warp the image perspective with the perspective transform calculated above
        img_warp = cv2.warpPerspective(img, M, (600,600))

        # create the 25 (5*5) subplots and put them in the list
        xdim = 5
        ydim = 5
        cur = 0
        onrow = 0
        oncol = 0

        cellsize = int(500 / xdim) # each cell is 96x96 pixels (480/5)

        for row in range(0, ydim):
            for col in range(0, xdim):
                cell = img_warp[(cellsize*row+50):(cellsize*row+cellsize+50), (cellsize*col+50):(cellsize*col+cellsize+50)]
                Image.fromarray(cell).save(directory+'output/out_'+str(i)+'_'+str(row)+'_'+str(col)+'.jpg')
