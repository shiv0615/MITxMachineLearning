import numpy as np
import common
import em

X = np.loadtxt(r"../Data/netflix/test_incomplete.txt")
X_gold = np.loadtxt(r"../Data/netflix/test_complete.txt")

K = 4
n, d = X.shape
seeds = 0
for ncluster in [K]:
    costs = []
    for seed in [seeds]:
        mixture, post = common.init(X, K=ncluster, seed=seed)
        mixture, post, cost = em.run(X=X, mixture=mixture, post=post)
        print(post)
        print(mixture)
        # common.plot(X, mixture=mixture, post=post, title='Init')
        costs.append(cost)
    print(f'nclusters:{ncluster}, seed:{seeds}, cost:{costs}')
