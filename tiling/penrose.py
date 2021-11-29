import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

class Kite():

    def __init__(self,point,angle):
        a1 = (np.pi/180.)*36.

        xs = [0,np.cos(a1),1.,np.cos(a1),0]
        ys = [0,np.sin(a1),0,-np.sin(a1),0]

        rmat = np.array([
            [np.cos(angle),(-1.)*np.sin(angle)],
            [np.sin(angle),np.cos(angle)]
        ])

        vs = np.vstack((xs,ys))
        vs = np.dot(vs.T,rmat).T
        vs[0]+=point[0]
        vs[1]+=point[1]
        self.verts = []
        for i in range(len(xs)): self.verts.append((vs[0][i],vs[1][i]))

        self.codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self.path = Path(self.verts,self.codes)

class Dart():

    def __init__(self,point,angle):
        a1 = (np.pi/180.)*36.
        phi = (1.+np.sqrt(5))/2.

        xs = [0,np.cos(a1),1./phi,np.cos(a1),0]
        ys = [0,np.sin(a1),0,-np.sin(a1),0]

        rmat = np.array([
            [np.cos(angle),(-1.)*np.sin(angle)],
            [np.sin(angle),np.cos(angle)]
        ])

        vs = np.vstack((xs,ys))
        vs = np.dot(vs.T,rmat).T
        vs[0]+=point[0]
        vs[1]+=point[1]
        self.verts = []
        for i in range(len(xs)): self.verts.append((vs[0][i],vs[1][i]))

        self.codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self.path = Path(self.verts,self.codes)

if __name__ == '__main__':
    phi = (1.+np.sqrt(5))/2.
    a1 = (np.pi/180.)*36.

    k1 = Kite([0,0],a1)
    d1 = Dart([1+(1./phi),0],np.pi)

    F = plt.figure()
    ax = F.add_subplot(111)
    ptc = patches.PathPatch(k1.path,fc='r',ec='gold',linewidth=3)
    ax.add_patch(ptc)
    ptc = patches.PathPatch(d1.path,fc='b',ec='gold',linewidth=3)
    ax.add_patch(ptc)
    ax.set_xlim(-.5,2)
    ax.set_ylim(-1,1)
    plt.show()