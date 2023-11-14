import ctypes
import math
import time
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from utils import logger
from collections import defaultdict
# from sklearn.cluster import KMeans


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


class NearbyPopularPlaces(object):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.distanceThreshold = None
        self.checkinMatrix = None
        self.visitedLids = {}
        self.poiCoos = None
        self.popularityFactor = {}
        self.activeUsers = {}

    def computeDistanceDistribution(self, checkinMatrix, poiCoos):
        distribution = defaultdict(int)
        all_distances = []
        for uid in range(checkinMatrix.shape[0]):
            lids = checkinMatrix[uid, :].nonzero()[0]
            for i in range(len(lids)):
                for j in range(i+1, len(lids)):
                    lid1, lid2 = lids[i], lids[j]
                    coo1, coo2 = poiCoos[lid1], poiCoos[lid2]
                    distance = int(dist(coo1, coo2))
                    distribution[distance] += 1
                    all_distances.append(distance)
        total = 1.0 * sum(distribution.values())
        for distance in distribution:
            distribution[distance] /= total
        distribution = sorted(distribution.items(), key=lambda k: k[0])
        all_distances = np.array(all_distances)
        self.distanceThreshold = np.quantile(all_distances, self.threshold)
        return zip(*distribution[1:])

    def calculatePopularities(self, checkinMatrix, poiCoos, poiCheckinCounts, activeUsers):
        self.checkinMatrix = checkinMatrix
        for uid in range(checkinMatrix.shape[0]):
            self.visitedLids[uid] = checkinMatrix[uid, :].nonzero()[0]

        startTime = time.time()
        logger('Fitting distances distribution ...')

        self.poiCoos = poiCoos
        x, t = self.computeDistanceDistribution(checkinMatrix, poiCoos)

        self.popularityFactor = defaultdict(int)
        for lid, row in poiCheckinCounts.iterrows():
            self.popularityFactor[lid] = np.log(row['checkins'])
        self.activeUsers = activeUsers[activeUsers.active].index

        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')

    def predict(self, uid, lj):
        if np.any([dist(self.poiCoos[li], self.poiCoos[lj]) < self.distanceThreshold for li in self.visitedLids[uid]]):
            if not (uid in self.activeUsers):
                return self.popularityFactor[lj]
        return 0.0


def nearby_predict(modelId, u):
    """
    Since the predict() method of this model is so slow, this function aids in
    parallelizing it.

    Given the Python object ID of the Power Law model and a user ID,
    compute the scores of all the POIs.
    """
    model = ctypes.cast(modelId, ctypes.py_object).value
    poisCount = model.checkinMatrix.shape[1]
    results = np.array([
        model.predict(u, l)
        for l in range(poisCount)
    ])
    return results