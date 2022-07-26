import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import imageio

import matplotlib as mpl

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

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

    def _make_line(self,xi,yi,xd=None,yd=None):
        if not xd:
            xs = np.array([self.xs[_] for _ in xi])
        else:
            xs = np.array([self.xs[_]+d*self.xs[2] for _,d in zip(xi,xd)])
        if not yd:
            ys = np.array([self.xs[_] for _ in yi])
        else:
            ys = np.array([self.xs[_]+d*self.xs[2] for _,d in zip(yi,yd)])
        vs = np.vstack((np.array(xs),np.array(ys)))
        vs = np.dot(vs.T,self.rmat).T.squeeze()
        
        return [
            np.array(vs[0])+self.origin[0],
            np.array(vs[1])+self.origin[1]
        ]
    
    def add_to_ax(self,ax,fc,lc):
        for tile in self.tiles:
            p = patches.PathPatch(tile,fc=fc,ec='None')
            ax.add_patch(p)
        for line in self.lines:
            ax.plot(*line,'-',c=lc,lw=1)

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
        xs = [self.xs[0],self.xs[0]+self.xs[2],self.xs[0]+self.xs[2],self.xs[0],self.xs[0]]
        ys = [self.xs[_] for _ in [1,1,2,2,1]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [self.xs[_] for _ in [1,2,2,3,3,2,2,1,1]]
        ys = [self.xs[_] for _ in [0,0,1,1,2,2,3,3,0]]
        self.tiles.append(self._make_path(xs,ys))
        self.lines.append(self._make_line([1,1],[0,3]))
        self.lines.append(self._make_line([2,2,3],[0,1,1]))
        self.lines.append(self._make_line([3,2,2],[2,2,3]))
        self.lines.append(self._make_line([0,0,0,0],[1,1,2,2],[0,1,1,0]))

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

class T5(TileBase):
    def __init__(self,L,x0,y0):
        super().__init__(L,x0,y0)
        self.build()

    def build(self):
        xs = [self.xs[_] for _ in [1,2,3,3,2,1,2,1,0,0]] + [self.xs[0]-self.xs[1],0] + [self.xs[1]]
        ys = [self.xs[_] for _ in [0,0,1,2,1,2,3,3,2,1]] + [0,self.xs[0]-self.xs[1]] + [self.xs[0]]
        self.tiles = [self._make_path(xs,ys)]
        self.lines.append(self._make_line([0,1],[2,3]))
        self.lines.append(self._make_line([2,3],[0,1]))
        self.lines.append(self._make_line([0,0,1,1],[1,1,0,0],[0,1,1,0],[0,1,1,0]))
        self.lines.append(self._make_line([2,1,2,3],[3,2,1,2]))

class T6(TileBase):
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
        xs = [self.xs[_] for _ in [1,2,3,3,1]]
        ys = [self.xs[_] for _ in [0,0,1,2,0]]
        self.tiles.append(self._make_path(xs,ys))
        xs = [self.xs[_] for _ in [0,2,1,0,0]]
        ys = [self.xs[_] for _ in [1,3,3,2,1]]
        self.tiles.append(self._make_path(xs,ys))
        self.lines.append(self._make_line([0,1],[2,3]))
        self.lines.append(self._make_line([2,3],[0,1]))
        self.lines.append(self._make_line([0,1],[1,0]))
        self.lines.append(self._make_line([2,3],[3,2]))
        self.lines.append(self._make_line([0,1,2,1,0],[1,0,1,2,1],[1,1,1,1,1],[1,1,1,1,1]))

class T7(TileBase):
    def __init__(self,L,x0,y0):
        super().__init__(L,x0,y0)
        self.build()

    def build(self):
        xs = [self.xs[_] for _ in [2,1,0,0,1,2]] + [self.xs[0]+self.xs[2],self.xs[2]] 
        ys = [self.xs[_] for _ in [3,3,2,1,0,0]] + [self.xs[1]+self.xs[2],self.xs[3]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [self.xs[3],self.xs[3],self.xs[2]*2,self.xs[3]]
        ys = [self.xs[1],self.xs[2],0,self.xs[1]]
        self.tiles.append(self._make_path(xs,ys))
        self.lines.append(self._make_line([0,1],[2,3]))
        self.lines.append(self._make_line([0,1],[1,0]))
        self.lines.append(self._make_line([2,0,2],[0,1,3],[0,1,0],[0,1,0]))
        self.lines.append(self._make_line([3,2,3],[1,1,2],[0,1,0],[0,1,0]))

L = 1

X,Y = 9,14
mf = 9
c1 = 'maroon'
c2 = 'tab:orange'
if X > Y:
    fx = mf
    fy = fx*(Y/X)
else:
    fy = mf
    fx = fy*(X/Y)
speed = 20
extras = 0

import copy

def make_new_column(Y,i,L=L):
    return [np.random.choice([T1,T2,T3,T4,T4,T4,T5,T5,T5,T6,T7])(L,i*L,j*L) for j in range(Y)]

def make_figure(tiles,tkeys,X,Y,x0,y0,L=L,c1=c1,c2=c2):
    F = plt.figure(figsize=(fx,fy),frameon=False)
    ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
    ax.set_aspect('auto')
    ax.set_axis_off()
    ax.fill_between([x0-L/2,x0+L*X-L/2],y0+L*Y-L/2,y0-L/2,color='w')
    F.add_axes(ax)
    for key in tkeys:
        for j,tile in enumerate(tiles[key]):
            tile.add_to_ax(ax,colorFader(c1,c2,j/(Y-1)),'k')
    ax.set_xlim(x0-L/2,x0+L*X-L/2)
    ax.set_ylim(y0-L/2,y0+L*Y-L/2)
    plt.savefig('frame.png')
    plt.close()

tiles = {}
tkeys = []
for i in range(X+1): 
    tiles[i] = []
for i in range(X+1):
    tkeys.append(i)
    for j in range(Y):
        _ = np.random.choice([T1,T2,T3,T4,T4,T4,T5,T5,T5,T6,T7])
        tt = _(L,i*L,j*L)
        tiles[i].append(tt)

images = []
for _i in range(X+extras):
    for x0 in np.linspace(_i*L,(_i+1)*L,speed)[:-1]:
        make_figure(tiles,tkeys,X,Y,x0,0)
        images.append(imageio.imread('frame.png'))
    i+=1
    tkeys = tkeys[1:]+[i]
    tiles[i] = make_new_column(Y,i)

for _j in range(X+1):
    i+=1
    tiles[i] = []
    for j,tile in enumerate(tiles[_j]):
        tile.origin = (i*L,j*L)
        tile.build()
        tiles[i].append(tile)
    for x0 in np.linspace((_i+_j+1)*L,((_i+_j)+2)*L,speed)[:-1]:
        make_figure(tiles,tkeys,X,Y,x0,0)
        images.append(imageio.imread('frame.png'))
    
    tkeys = tkeys[1:] + [i]
    

imageio.mimsave('./truch.mp4',images,fps=15)