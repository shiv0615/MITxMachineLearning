import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt

def plot_toy_data():
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.title('Scatter Plot: ToyData')
    plt.show()

X = np.loadtxt(r"netflix_incomplete.txt")
Xgold = np.loadtxt(r"netflix_complete.txt")
print(f'Shape of X: {X.shape}')
nclusters = [12]
seeds = [0,1,2,3,4]
for ncluster in nclusters:
    costs = []
    bics = []
    rmse = []
    for seed in seeds:
        print(f'ncluster: {ncluster}, seed: {seed}')
        mixture, post = common.init(X, K=ncluster, seed=seed)
        # mixture, post, cost = kmeans.run(X=X,mixture=mixture,post=post)
        # mixture, post, cost = naive_em.run(X=X, mixture=mixture, post=post)
        mixture, post, cost = em.run(X=X, mixture=mixture, post=post)
        Xnew = em.fill_matrix(X,mixture)
        bic = common.bic(X, mixture, cost)
        # common.plot(X, mixture=mixture, post=post, title='Init')
        costs.append(cost)
        bics.append(bic)
        rmse.append(common.rmse(Xgold,Xnew))
    print(f'nclusters:{ncluster}, seed:{seeds}, costs:{costs}, bics:{bics}, rmse:{rmse}')

