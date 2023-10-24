import math
import numpy as np


def dist(loc1, loc2):
    lat1, long1 = loc1[0], loc1[1]
    lat2, long2 = loc2[0], loc2[1]
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degreesToRadians = math.pi/180.0
    phi1 = (90.0 - lat1)*degreesToRadians
    phi2 = (90.0 - lat2)*degreesToRadians
    theta1 = long1*degreesToRadians
    theta2 = long2*degreesToRadians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)
    earthRadius = 6371
    return arc * earthRadius


def medianDistance(userCoos: tuple, predicted: list, poiCoos: list):
    """
    Given some average coordinate of a user, calculate the median distance of
    the recommended POIs to that user
    """

    distances = np.zeros(len(predicted))
    for i, poi in enumerate(predicted):
        distances[i] = dist(userCoos, poiCoos[poi])

    return np.median(distances)