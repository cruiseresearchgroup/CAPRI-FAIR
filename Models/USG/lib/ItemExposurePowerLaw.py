import math
import time
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from utils import logger
from collections import defaultdict


class ItemExposurePowerLaw(object):
    def __init__(self, a=None, b=None, alpha=10.0, exposureModel='Linear'):
        self.alpha = alpha
        self.a = a
        self.b = b
        self.poiCheckinCounts = None
        self.powerLaw = None
        self.exposureModel = exposureModel

    def fitExposureDistribution(self, poiCheckinCounts):
        self.poiCheckinCounts = poiCheckinCounts
        xs = np.arange(0, poiCheckinCounts['checkins'].max(), 10)
        freqs, bins = np.histogram(poiCheckinCounts['checkins'], bins=xs)
        bins = bins[1:]
        nonzero = (freqs != 0)

        startTime = time.time()

        if self.exposureModel == 'PowerLaw':
            logger('Fitting distances distribution (power law)...')
            x, t = np.log10(bins[nonzero]), np.log10(freqs[nonzero])
            self.powerLaw = Ridge(alpha=self.alpha).fit(x[:, np.newaxis], t)
            self.a, self.b = 10**self.powerLaw.intercept_, self.powerLaw.coef_[0]
            logger(f"Ridge, alpha={self.alpha:.1f}: y = {self.a:.2f} * x^{self.b:.2f}")
        elif self.exposureModel == 'Linear':
            logger('Fitting distances distribution (linear)...')
            x, t = bins[nonzero], freqs[nonzero]
            self.powerLaw = Ridge(alpha=self.alpha).fit(x[:, np.newaxis], t)
            self.a, self.b = self.powerLaw.intercept_, self.powerLaw.coef_[0]
            logger(f"Ridge, alpha={self.alpha:.1f}: y = {self.a:.2f} + {self.b:.2f}*x")
        elif self.exposureModel == 'Logistic':
            logger('Fitting distances distribution (logistic)...')
            x, t = bins[nonzero], freqs[nonzero]
            self.powerLaw = LogisticRegression(penalty=None).fit(x[:, np.newaxis], t)
            logger(f"Logistic, no L1")

        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')

    def predict(self, poiCount):
        """
        The value is independent of the user at this point in time,
        so to speed things up....

        int -> (poiCount,)
        """
        poisList = np.array(range(poiCount))
        x = self.poiCheckinCounts[['checkins']].reindex(poisList).fillna(1).values

        if self.exposureModel == 'PowerLaw':
            return np.log(self.a * (x ** self.b))
        elif self.exposureModel == 'Linear':
            return self.a + (x * self.b)
        elif self.exposureModel == 'Logistic':
            return self.powerLaw.predict(x)
