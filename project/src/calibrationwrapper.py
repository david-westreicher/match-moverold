'''
Created on 31.10.2012

@author: David
'''
from cv2 import cv
import numpy
import math
import calibration
import util

WINDOW_NAME = "Calibrate"
WINDOW_SIZE = 1100.0
BOX_NUM = 5
BOX_CORNERS = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
DEFAULT_FONT = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.3, 0.3, 0, 1, 8)
# should be lower than the minimum distance between 2 corners in the images (magic number, but works for the example images)
MIN_DISTANCE_THRESHOLD = 30
numpy.set_printoptions(suppress=True, edgeitems=12, linewidth=200)

def init():
    global currentFace, selectedCorners, correspondences
    # the face (left=0,right=1,bottom=2) which is currently defined
    currentFace = 0
    # the selected corners [(x1,y1),(x2,y2),...] of the current face
    selectedCorners = []
    # the 2d to 3d correspondences [((x,y),(x',y',z')),...]
    correspondences = []

# starts the calibration process
def calibrate(calibrationImageFile):
    global ratio
    init()
    
    # load the image and resize it to WINDOW_SIZE
    print("loading " + calibrationImageFile)
    windowImage = cv.LoadImage(calibrationImageFile)
    imgsize = cv.GetSize(windowImage)
    ratio = WINDOW_SIZE / imgsize[0]
    resizedImage = cv.CreateImage((int(WINDOW_SIZE), int(WINDOW_SIZE * imgsize[1] / imgsize[0])), windowImage.depth, windowImage.nChannels)
    cv.Resize(windowImage, resizedImage)
    windowImage = resizedImage
    
    # find the corners and paint them into the image
    corners = findCorners(calibrationImageFile)
    for corner in corners:
        cv.Circle(windowImage, (int(corner[0] * ratio), int(corner[1] * ratio)), 2, cv.RGB(0, 255, 0))
    
    # setup gui
    cv.NamedWindow(WINDOW_NAME, cv.CV_WINDOW_AUTOSIZE)
    cv.ShowImage(WINDOW_NAME, windowImage)
    cv.SetMouseCallback(WINDOW_NAME, onMouseClick, (windowImage, corners))
    onMouseClick(1, 347 * WINDOW_SIZE / 1100.0, 252 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 366 * WINDOW_SIZE / 1100.0, 445 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 527 * WINDOW_SIZE / 1100.0, 358 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 522 * WINDOW_SIZE / 1100.0, 173 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 729 * WINDOW_SIZE / 1100.0, 246 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 554 * WINDOW_SIZE / 1100.0, 166 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 558 * WINDOW_SIZE / 1100.0, 357 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 712 * WINDOW_SIZE / 1100.0, 447 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 535 * WINDOW_SIZE / 1100.0, 577 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 384 * WINDOW_SIZE / 1100.0, 476 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 542 * WINDOW_SIZE / 1100.0, 387 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    onMouseClick(1, 696 * WINDOW_SIZE / 1100.0, 475 * WINDOW_SIZE / 1100.0, None, (windowImage, corners))
    cv.WaitKey()
    
# find corners [(x1,y1),(x2,y2),...] iteratively (min. 300) and refine for sub-pixel accuracy
def findCorners(calibrationImageFile):
    image = cv.LoadImage(calibrationImageFile, cv.CV_LOAD_IMAGE_GRAYSCALE)
    cornerMap = cv.CreateMat(image.height, image.width, cv.CV_32FC1)
    print("starting harris corner detector")
    cv.CornerHarris(image, cornerMap, 2)
    harrisThreshold = 2000.0
    corners = []
    while len(corners) < 300 and harrisThreshold <= 64000:
        corners = []
        print("finding corners with threshold: " + str(harrisThreshold))
        invThreshold = 1 / harrisThreshold
        for y in range(0, image.height):
            for x in range(0, image.width):
                harris = cv.Get2D(cornerMap, y, x)
                if harris[0] > invThreshold:
                    corners.append((x, y))
        print("found " + str(len(corners)) + " corners")
        harrisThreshold *= 2
    # refine corner positions with sub-pixel accuracy
    corners = cv.FindCornerSubPix(image, corners, (5, 5), (int(-1), int(-1)), (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 20, 0.03))
    return corners

def reproject(p, correspondences, windowImage):
    for face in range(0, 3):
        for x in range(1, BOX_NUM * 2 + 1):
            for y in range(1, BOX_NUM * 2 + 1):
                coord3d = create3dCoord(face, x , y)
                projectedCorner = numpy.dot(p , numpy.array([coord3d[0], coord3d[1], coord3d[2], 1]))
                projectedCorner /= projectedCorner[2]
                intPos = (int(projectedCorner[0] * ratio), int(projectedCorner[1] * ratio))
                cv.Circle(windowImage, intPos , 4, cv.RGB(255, 0, 0))
    projectionerr = 0
    squaredprojectionerr = 0
    for (x2d, x3d) in correspondences:
        projectedPos = numpy.dot(p , numpy.array([x3d[0], x3d[1], x3d[2], 1]))
        projectedPos /= projectedPos[2]
        dst = util.distance((projectedPos[0], projectedPos[1]), x2d)
        projectionerr += dst
        squaredprojectionerr += dst ** 2
    print("mean reprojection error: " + str(projectionerr / len(correspondences)) + "px")
    print("root mean squared reprojection error: " + str(math.sqrt(squaredprojectionerr / len(correspondences))) + "px")


def onMouseClick(event, x, y, flags, (windowImage, corners)):
    global selectedCorners, currentFace
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print("\tonMouseClick(1," + str(x) + "* WINDOW_SIZE / 1100.0, " + str(y) + "* WINDOW_SIZE / 1100.0, None, (windowImage, corners))")
        selectedCorners.append((x / ratio, y / ratio))
        if len(selectedCorners) == 4:
            # if a face is defined add the correspondences
            correspondences.extend(getCorrespondencesForFace(selectedCorners, corners, currentFace, windowImage))
            selectedCorners = []
            currentFace += 1
            if currentFace == 3:
                # if all 3 faces are defined calculate the camera parameters, print them and reproject
                p, c, k, r = calibration.calculateCameraParameters(correspondences)
                print("camera centre C:")
                print(c)
                print("calibration matrix K:")
                print(k)
                print("rotation matrix R:")
                print(r)
                reproject(p, correspondences, windowImage)
                cv.SaveImage("../../out.bmp", windowImage)
        cv.Circle(windowImage, (int(x), int(y)), 8, cv.RGB(255, 0, 0))
        cv.ShowImage(WINDOW_NAME, windowImage)
    
# computes the 3d correspondences from 4 selected corners (which define the face) and the corners found from the corner detection
def getCorrespondencesForFace(selectedCorners, corners, currentFace, windowImage):
    h = compute2DHomography(selectedCorners)
    correspondences = []
    # for every possible corner in the face
    for x in range(0, BOX_NUM * 2):
        for y in range(0, BOX_NUM * 2):
            # project the possible corner (in the range [-1,1]x[-1,1]) into image space with the homography h
            scale = (BOX_NUM * 2 - 1.0) / 2.0
            projectedCorner = numpy.dot(h , numpy.array([x / scale - 1.0, y / scale - 1.0, 1]))
            projectedCorner /= projectedCorner[2]
            
            # find the closest corner from the corner detection set
            minDistance = 10000
            minCorner = None
            pos = (projectedCorner[0], projectedCorner[1])
            for corner in corners:
                dist = util.distance(corner, pos)
                if dist < minDistance:
                    minDistance = dist
                    minCorner = corner
            
            # check if the closest corner belongs to this corner -> add correspondence and paint a circle + the coordinates
            if minDistance < MIN_DISTANCE_THRESHOLD:
                coord3d = create3dCoord(currentFace, x + 1, y + 1)
                intPos = (int(pos[0] * ratio), int(pos[1] * ratio))
                cv.Circle(windowImage, intPos , 6, cv.RGB(0, 0, 255))
                # cv.PutText(windowImage, str(coord3d), intPos, DEFAULT_FONT , cv.RGB(0, 0, 255))
                correspondences.append((minCorner, coord3d))
            
    print("found " + str(len(correspondences)) + " correspondences")
    return correspondences


# find the 2d homography "h" between the BOX_CORNERS and the selected corners: (x,y,w)=H*(boxX,boxY,boxW)
def compute2DHomography(selectedCorners):
    matrixList = []
    for i in range(0, 4):
        x = BOX_CORNERS[i]
        xprime = selectedCorners[i]
        matrixList.append([0, 0, 0, -x[0], -x[1], -1, xprime[1] * x[0], xprime[1] * x[1], xprime[1] * 1])
        matrixList.append([x[0], x[1], 1, 0, 0, 0, -xprime[0] * x[0], -xprime[0] * x[1], -xprime[0] * 1])
    matrix = numpy.array(matrixList)
    _, _, v = numpy.linalg.svd(matrix)
    hcolumn = v[ -1, : ]
    h = numpy.reshape(hcolumn, (3, 3))
    return h

# creates a 3d coordinate from 2 coordinates depending on the face (left,right,bottom)
def create3dCoord(currentFace, x, y):
    if currentFace == 0:
        return (x, y, 0)
    elif currentFace == 1:
        return (0, x, y)
    else:
        return (x, 0, y)
