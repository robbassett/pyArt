import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

class TileBase():
    def __init__(self,L,x0,y0):
        self.xs = np.linspace(-L/2,L/2,4)
        self.rot = np.random.randint(0,3,1)*np.pi/2
        self.rmat = np.array([
            [np.cos(self.rot),(-1.)*np.sin(self.rot)],
            [np.sin(self.rot),np.cos(self.rot)]
        ])
        self.origin = (x0,y0)
        self.tiles = []
        self.lines = []

    def _make_path(self,xs,ys):

        vs = np.vstack((np.array(xs),np.array(ys)))
        vs = np.dot(vs.T,self.rmat).T.squeeze()

        vs[0]+=self.origin[0]
        vs[1]+=self.origin[1]
        verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]

        codes = [Path.MOVETO] + [Path.LINETO]*(len(xs)-2) + [Path.CLOSEPOLY]

        return Path(verts,codes)

    def _make_line(self,xi,yi,xd=None):
        xs = np.array([self.xs[_] for _ in xi])
        ys = np.array([self.xs[_] for _ in yi])
        vs = np.vstack((np.array(xs),np.array(ys)))
        vs = np.dot(vs.T,self.rmat).T.squeeze()
        
        return [
            np.array(vs[0])+self.origin[0],
            np.array(vs[1])+self.origin[1]
        ]

class T1(TileBase):
    def __init__(self,L,x0,y0):
        super().__init__(L,x0,y0)
        self.build()

    def build(self):
        xs = [self.xs[_] for _ in [1,2,2,3,3,2,2,1,1,0,0,1,1]]
        ys = [self.xs[_] for _ in [0,0,1,1,2,2,3,3,2,2,1,1,0]]
        self.tiles = [self._make_path(xs,ys)]
        self.lines.append(self._make_line([0,1,1],[1,1,0]))
        self.lines.append(self._make_line([2,2,3],[0,1,1]))
        self.lines.append(self._make_line([3,2,2],[2,2,3]))
        self.lines.append(self._make_line([1,1,0],[3,2,2]))

class T2(TileBase):
    def __init__(self,L,x0,y0):
        super().__init__(L,x0,y0)
        self.build()

    def build(self):
        xs = [self.xs[_] for _ in [1,2,2,3,3,2,1,1]]
        ys = [self.xs[_] for _ in [0,0,1,1,2,2,1,0]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [self.xs[_] for _ in [0,2,1,0,0]]
        ys = [self.xs[_] for _ in [1,3,3,2,1]]
        self.tiles.append(self._make_path(xs,ys))
        self.lines.append(self._make_line([1,1,2,3],[0,1,2,2]))
        self.lines.append(self._make_line([2,2,3],[0,1,1]))
        self.lines.append(self._make_line([0,2],[1,3]))
        self.lines.append(self._make_line([0,1],[2,3]))
           

class T3(TileBase):
    def __init__(self,L,x0,y0):
        super().__init__(L,x0,y0)
        self.build()

    def build(self):
        xs = [self.xs[0],self.xs[0]-self.xs[1]/1.5,self.xs[0]-self.xs[1]/1.5,self.xs[0],self.xs[0]]
        ys = [self.xs[_] for _ in [1,1,2,2,1]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [self.xs[_] for _ in [1,2,2,3,3,2,2,1,1]]
        ys = [self.xs[_] for _ in [0,0,1,1,2,2,3,3,0]]
        self.tiles.append(self._make_path(xs,ys))
        self.lines.append(self._make_line([1,1],[0,3]))
        self.lines.append(self._make_line([2,2,3],[0,1,1]))
        self.lines.append(self._make_line([3,2,2],[2,2,3]))
        self.lines.append(self._make_line([0,1,1,0],[1,1,2,2]))

class T4(TileBase):
    def __init__(self,L,x0,y0):
        super().__init__(L,x0,y0)
        self.build()

    def build(self):
        xs = [self.xs[_] for _ in [1,2,0,0,1]]
        ys = [self.xs[_] for _ in [0,0,2,1,0]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [self.xs[_] for _ in [3,3,2,1,3]]
        ys = [self.xs[_] for _ in [1,2,3,3,1]]
        self.tiles.append(self._make_path(xs,ys))
        self.lines.append(self._make_line([0,1],[1,0]))
        self.lines.append(self._make_line([0,2],[2,0]))
        self.lines.append(self._make_line([1,3],[3,1]))
        self.lines.append(self._make_line([2,3],[3,2]))

L = 1

F = plt.figure(figsize=(6,9))
ax = F.add_subplot(111)
for i in range(24):
    for j in range(36):
        _ = np.random.choice([T1,T2,T4])
        _ = T4
        tt = _(L,i*L,j*L)
        for tile in tt.tiles:
            p = patches.PathPatch(tile,fc='r',ec='None')
            ax.add_patch(p)
        for line in tt.lines:
            ax.plot(*line,'k-',lw=2)
ax.set_xlim(-L/2,L*24-L/2)
ax.set_ylim(-L/2,L*36-L/2)
plt.show()