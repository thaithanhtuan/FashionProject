import ipdb
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

def makeT(cp):
    # cp: [K x 2] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]
    T = np.zeros((K+3, K+3))
    T[:K, 0] = 1
    #print("T:", T)
    T[:K, 1:3] = cp
    #print("T:", T)
    T[K, 3:] = 1
    #print("T:", T)
    T[K+1:, 3:] = cp.T
    #print("T:", T)
    R = squareform(pdist(cp, metric='euclidean'))
    #print("R:", R)
    R = R * R
    #print("R:", R)
    R[R == 0] = 1 # a trick to make R ln(R) 0
    #print("R:", R)
    R = R * np.log(R)
    #print("R:", R)
    np.fill_diagonal(R, 0)
    #print("R:", R)
    T[:K, 3:] = R
    #print("T:", T)
    return T

def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    #print("N:",N,",K:",K)
    pLift = np.zeros((N, K+3))
    pLift[:,0] = 1
    pLift[:,1:3] = p
    #print("pList:",pLift)
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:,3:] = R
    return pLift

# source control points
x, y = np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)
x, y = np.meshgrid(x, y)
xs = x.flatten()
ys = y.flatten()
cps = np.vstack([xs, ys]).T

# target control points
xt = xs + np.random.uniform(-0.3, 0.3, size=xs.size)
yt = ys + np.random.uniform(-0.3, 0.3, size=ys.size)

# construct T
T = makeT(cps)

# solve cx, cy (coefficients for x and y)
xtAug = np.concatenate([xt, np.zeros(3)])
ytAug = np.concatenate([yt, np.zeros(3)])
cx = nl.solve(T, xtAug) # [K+3]
cy = nl.solve(T, ytAug)
print("cx shape:", cx)
print("yx shape:", cy)

# dense grid
N = 30
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
x, y = np.meshgrid(x, y)
xgs, ygs = x.flatten(), y.flatten()
gps = np.vstack([xgs, ygs]).T

# transform
pgLift = liftPts(gps, cps) # [N x (K+3)]
xgt = np.dot(pgLift, cx.T)
ygt = np.dot(pgLift, cy.T)

# display
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.subplot(1, 2, 1)
plt.title('Source')
plt.grid()
plt.scatter(xs, ys, marker='+', c='b', s=40)
plt.scatter(xgs, ygs, marker='.', c='r', s=5)
plt.subplot(1, 2, 2)
plt.title('Target')
plt.grid()
plt.scatter(xt, yt, marker='+', c='r', s=40)
plt.scatter(xgt, ygt, marker='.', c='b', s=1)
plt.show()