'''
Created on 01.11.2012

@author: David
'''
import numpy
import math

# Gold Standard Algorithm for estimating P (Multiple View Geometry Sec. Edition, page:181, (7.1))
# returns (p, camera center, calibration matrix, rotation matrix)
def calculateCameraParameters(correspondences):
    correspondences, t, u = normalize(correspondences)
    p = dlt(correspondences)
    p = nonLinearOptimization(p, correspondences)
    p = denormalize(p, t, u)
    c, k, r = extractCameraParameters(p)
    return p, c, k, r

# construct the matrix A for the estimation of P (Multiple View Geometry Sec. Edition, page:179, (7.2))
def constructMatrixA(correspondences):
    matrixList = []
    for (x, bigX) in correspondences:
        matrixList.append([0, 0, 0, 0, -bigX[0], -bigX[1], -bigX[2], -1, x[1] * bigX[0], x[1] * bigX[1], x[1] * bigX[2], x[1] * 1])
        matrixList.append([bigX[0], bigX[1], bigX[2], 1, 0, 0, 0, 0, -x[0] * bigX[0], -x[0] * bigX[1], -x[0] * bigX[2], -x[0] * 1])
    return numpy.matrix(matrixList)


# direct linear transform of the correspondences
def dlt(correspondences):
    matrix = constructMatrixA(correspondences)
    _, _, v = numpy.linalg.svd(matrix)
    # take the last column
    pcolumn = v[-1, : ]
    p = numpy.reshape(pcolumn, (3, 4))
    print("dlt-estimation of P:")
    print(p)
    return p

# normalize the correspondences (mean origin and mean length)
# returns the normalized correspondences and the transformation matrices (t,u)
def normalize(correspondences):
    # transform correspondences from tuples into numpy.array's
    corr = []
    for (imageCoord, worldCoord) in correspondences:
        corr.append((numpy.array(imageCoord), numpy.array(worldCoord)))
        
    # compute mean origin
    imageOrigin = numpy.array([0.0, 0.0])
    worldOrigin = numpy.array([0.0, 0.0, 0.0])
    for (imageCoord, worldCoord) in corr:
        imageOrigin += imageCoord
        worldOrigin += worldCoord
    imageOrigin /= len(correspondences)
    worldOrigin /= len(correspondences)
    
    # compute mean norm
    imageNorm = 0.0
    worldNorm = 0.0
    for (imageCoord, worldCoord) in corr:
        imageNorm += numpy.linalg.norm(imageOrigin - imageCoord)
        worldNorm += numpy.linalg.norm(worldOrigin - worldCoord)
    imageNorm /= len(correspondences)
    worldNorm /= len(correspondences)
    tscale = math.sqrt(2) / imageNorm
    uscale = math.sqrt(3) / worldNorm
    
    # create normalized correspondences by multiplying with matrix t, u
    t = numpy.array([[tscale, 0, -imageOrigin[0] * tscale ], [0, tscale, -imageOrigin[1] * tscale ], [0, 0, 1]])
    u = numpy.array([[uscale, 0, 0, -worldOrigin[0] * uscale ], [0, uscale, 0, -worldOrigin[1] * uscale ], [0, 0, uscale, -worldOrigin[2] * uscale], [0, 0, 0, 1]])
    normalizedCorrespondences = []
    for (imageCoord, worldCoord) in corr:
        normalizedImageCoord = numpy.dot(t, numpy.append(imageCoord, [1]))
        normalizedWorldCoord = numpy.dot(u, numpy.append(worldCoord, [1]))
        normalizedCorrespondences.append((normalizedImageCoord, normalizedWorldCoord))
    return normalizedCorrespondences, t, u

def denormalize(p, t, u):
    return numpy.dot(numpy.dot(numpy.linalg.inv(t), p), u)


# extract the camera paramters from p 
# returns (camera center, calibration matrix, rotation matrix)
def extractCameraParameters(p):
    # calculate camera centre c = -A^(-1)*b, P = [A|b]
    a = p[0:3, 0:3]
    b = p[:, -1]
    c = -numpy.dot(numpy.linalg.inv(a), b)
    # rq decomposition
    k, r = rq(a)
    return c, k, r


def nonLinearOptimization(p, correspondences):
    return p


# rq-decomposition using qr-decomposition taken from -> http://www.janeriksolem.net/2011/03/rq-factorization-of-camera-matrices.html
def rq(A):
    Q, R = numpy.linalg.qr(numpy.flipud(A).T) 
    R = numpy.flipud(R.T)
    Q = Q.T 
    return R[:, ::-1], Q[::-1, :]
