import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

class SquareP(object):

    def __init__(self,cent,L=1.,rang=0.):
        h = L/2.
        xs = (np.array([-1.,1.,1.,-1.,-1.])*h)
        ys = (np.array([-1.,-1.,1.,1.,-1.])*h)
        
        rmat = np.array([
            [np.cos(rang),(-1.)*np.sin(rang)],
            [np.sin(rang),np.cos(rang)]
        ])

        vs = np.vstack((xs,ys))
        vs = np.dot(vs.T,rmat).T
        vs[0]+=cent[0]
        vs[1]+=cent[1]
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

class EqTri(object):
    
    def __init__(self,cent,L=1.,rang=0.):
        h=L/2.
        t1 = np.tan(np.pi/6.)
        c1 = np.cos(np.pi/6.)
        xs = (np.array([-1.,1.,0,1.])*h)
        ys = (np.array([(-1.)*t1,(-1.)*t1,1./c1,(-1.)*t1])*h)
        vs = np.vstack((xs,ys))
        
        rmat = np.array([
            [np.cos(rang),(-1.)*np.sin(rang)],
            [np.sin(rang),np.cos(rang)]
        ])

        vs = np.dot(vs.T,rmat).T
        vs[0]+=cent[0]
        vs[1]+=cent[1]
        self.verts = []
        for i in range(len(xs)): self.verts.append((vs[0][i],vs[1][i]))
        self.codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self.path = Path(self.verts,self.codes)

