import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

class P2_base():

    def __init__(self,point,angle,l):
        self.L = l
        self.angle = angle
        self.point = point
        self.a1 = (np.pi/180.)*36.
        self.phi = l*(1.+np.sqrt(5))/2.

    def _make_path(self):
        rmat = np.array([
            [np.cos(self.angle),(-1.)*np.sin(self.angle)],
            [np.sin(self.angle),np.cos(self.angle)]
        ])

        vs = np.vstack((self.xs,self.ys))
        vs = np.dot(vs.T,rmat).T
        vs[0]+=self.point[0]
        vs[1]+=self.point[1]
        self.verts = [(vs[0][i],vs[1][i]) for i in range(len(self.xs))]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self.path = Path(self.verts,codes)

class Kite(P2_base):

    def __init__(self,point,angle,l):
        super().__init__(point,angle,l)
        self.type='Kite'
        self.xs = [0,l*np.cos(self.a1),l,l*np.cos(self.a1),0]
        self.ys = [0,l*np.sin(self.a1),0,-l*np.sin(self.a1),0]
        self._make_path()

    def inflate(self):
        l = self.L - self.phi
        k1 = Kite(self.verts[1],self.angle-2*self.a1,l)
        k2 = Kite(self.verts[3],self.angle+2*self.a1,l)
        d1 = Dart(self.verts[0],self.angle+4*self.a1,l)
        d2 = Dart(self.verts[0],self.angle-4*self.a1,l)

        return k1,k2,d1,d2

class Dart(P2_base):

    def __init__(self,point,angle,l):
        super().__init__(point,angle,l)
        self.type='Dart'
        self.xs = [0,l*np.cos(self.a1),l*l/self.phi,l*np.cos(self.a1),0]
        self.ys = [0,l*np.sin(self.a1),0,-l*np.sin(self.a1),0]
        self._make_path()

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
        cs = ['tab:blue','royalblue','dodgerblue'] if t.type == 'Kite' else ['tab:orange','darkorange','orange']
        p = patches.PathPatch(t.path,fc=np.random.choice(cs),ec='k')
        ax.add_patch(p)
    ax.set_xlim(-.52,.11)
    ax.set_ylim(.232,.65)
    plt.show()