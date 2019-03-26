# coding: utf-8

import numpy as np

CLUSTER_NUMBER = 5
MAX_ITERATION = 100


def KMeans(sets):
    # init
    cluster_ids = np.random.randint(0, CLUSTER_NUMBER, sets.shape[0])
    cogs = np.empty((CLUSTER_NUMBER, sets.shape[1]))

    # kmeans
    for i in range(MAX_ITERATION):
        for k in range(CLUSTER_NUMBER):
            if np.sum(cluster_ids == k) > 0:
                cogs[k] = np.mean(sets[cluster_ids == k], axis=0)

        new_ids = np.array([np.argmin(np.sum(abs((data - cogs)), axis=1)) for data in sets])

        if np.all(cluster_ids == new_ids):
            break
        else:
            cluster_ids = new_ids

    return cluster_ids, cogs


if __name__ == '__main__':
    question = np.random.rand(100, 2)
    a, b = KMeans(question)
    print(a)
    print(b)
