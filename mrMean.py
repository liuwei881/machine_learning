#coding=utf-8


from mrjob.job import MRJob
from numpy import *

class MRmean(MRJob):
    def __init__(self, *args, **kwargs):
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    def map(self, key, val):
        if False:
            yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal ** 2

    def map_final(self):
        mn = self.inSum/self.inCount
        mnSq = self.inSqSum/self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedValues):
        cumVal = 0.0
        cumSumSq = 0.0
        cumN = 0.0
        for valArr in packedValues:
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal/cumN
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean)/cumN
        yield (mean, var)

    def steps(self):
        return ([self.mr(mapper=self.map, reducer=self.reduce, mapper_final=self.map_final)])


# pegasos算法
def predict(w, x):
    return w*x.T


def batchPegasos(dataSet, labels, lam, T, k):
    m, n = shape(dataSet)
    dataIndex = range(m)
    for t in range(1, T+1):
        wDelta = mat(zeros(n))
        eta = 1.0/(lam*t)
        random.shuffle(dataIndex)
        for j in range(k):
            i = dataIndex[j]
            p = predict(w, dataSet[i, :])
            if labels[i]*p < 1:
                wDelta += labels[i]*dataSet[i, :].A
        w = (1.0 - 1/t)*w + (eta/k)*wDelta
    return w


if __name__ == '__main__':
    MRmean.run()