'''
Created on 01.11.2012

@author: David
'''
import math
import numpy
from Quaternion import Quat
import Quaternion

# computes the 2d distance between vector a and b
def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

_EPS = numpy.finfo(float).eps * 4.0

    
def quaternion_to_matrix(quatArr):
    quatArr = Quaternion.normalize(quatArr)
    q = Quat(attitude=quatArr)
    return q.transform


def quaternion_from_matrix(matrix):
    q = Quat(attitude=matrix)
    print(q.q)
    Quaternion.normalize(q.q)
    print(q.q)
    return q
