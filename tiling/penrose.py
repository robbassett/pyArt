import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

class Kite():

    def __init__(self,point,angle,l):
        self.type='Kite'
        self.L = l
        a1 = (np.pi/180.)*36.
        phi = l*(1.+np.sqrt(5))/2.

        xs = [0,l*np.cos(a1),l,l*np.cos(a1),0]
        ys = [0,l*np.sin(a1),0,-l*np.sin(a1),0]

        rmat = np.array([
            [np.cos(angle),(-1.)*np.sin(angle)],
            [np.sin(angle),np.cos(angle)]
        ])

        vs = np.vstack((xs,ys))
        vs = np.dot(vs.T,rmat).T
        vs[0]+=point[0]
        vs[1]+=point[1]
        self.verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]
        self.angle = angle
        self.a1 = a1
        self.phi = phi

        self.codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self.path = Path(self.verts,self.codes)

    def inflate(self):
        l = self.L - self.phi
        k1 = Kite(self.verts[1],self.angle-2*self.a1,l)
        k2 = Kite(self.verts[3],self.angle+2*self.a1,l)
        d1 = Dart(self.verts[0],self.angle+4*self.a1,l)
        d2 = Dart(self.verts[0],self.angle-4*self.a1,l)

        return k1,k2,d1,d2

class Dart():

    def __init__(self,point,angle,l):
        self.type='Dart'
        self.L = l
        a1 = (np.pi/180.)*36.
        phi = l*(1.+np.sqrt(5))/2.

        xs = [0,l*np.cos(a1),l*l/phi,l*np.cos(a1),0]
        ys = [0,l*np.sin(a1),0,-l*np.sin(a1),0]

        rmat = np.array([
            [np.cos(angle),(-1.)*np.sin(angle)],
            [np.sin(angle),np.cos(angle)]
        ])

        vs = np.vstack((xs,ys))
        vs = np.dot(vs.T,rmat).T
        vs[0]+=point[0]
        vs[1]+=point[1]
        self.verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]
        self.angle = angle
        self.a1 = a1
        self.phi = phi

        self.codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self.path = Path(self.verts,self.codes)

    def inflate(self):
        l = self.L - self.phi
        k = Kite(self.verts[0],self.angle+np.pi,l)
        d1 = Dart(self.verts[1],self.angle-self.a1,l)
        d2 = Dart(self.verts[3],self.angle+self.a1,l)
        return k,d1,d2

        

if __name__ == '__main__':

    tiles = [Dart([0,0],-1.2*np.pi/2,1)]

    while len(tiles) < 5000:
        tmt = []
        for t in tiles: tmt+=list(t.inflate())
        tiles = tmt

    F = plt.figure()
    ax = F.add_subplot(111)
    ax.set_aspect('equal')
    for t in tiles:
        c = 'tab:blue' if t.type == 'Kite' else 'tab:purple'
        p = patches.PathPatch(t.path,fc=c,ec='k')
        ax.add_patch(p)
    ax.set_xlim(-.75,.5)
    ax.set_ylim(0,1)
    plt.show()