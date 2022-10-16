import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import imageio

import matplotlib as mpl

from truchet import TileBase,T1,T2,T3,T4,T5,T6,T7

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

# PAIR 1
class TRT1(TileBase):
    def __init__(self,L,x0,y0,flp=False):
        super().__init__(L,x0,y0,pf=0,flp=flp)
        self.build()

    def build(self):
        hxs = np.linspace(-self.L/2,self.L/2,7)
        xs = [hxs[_] for _ in [0,1,0,2,4,2,3,0,0]]
        ys = [hxs[_] for _ in [2,3,4,6,6,4,3,0,2]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [self.xs[_] for _ in [1,2,3,3,1]]
        ys = [self.xs[_] for _ in [0,0,1,2,0]]
        self.tiles.append(self._make_path(xs,ys))

class TRB1(TileBase):
    def __init__(self,L,x0,y0,flp=False):
        super().__init__(L,x0,y0,pf=0,flp=flp)
        self.build()

    def build(self):
        hxs = np.linspace(-self.L/2,self.L/2,7)
        xs = [hxs[_] for _ in [0,1,2,4,1,0,0]]
        ys = [hxs[_] for _ in [6,5,6,6,3,4,6]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [self.xs[_] for _ in [1,2,3,3,1]]
        ys = [self.xs[_] for _ in [0,0,1,2,0]]
        self.tiles.append(self._make_path(xs,ys))

# PAIR 2
class TRT2(TileBase):
    def __init__(self,L,x0,y0,flp=False):
        super().__init__(L,x0,y0,pf=0,flp=flp)
        self.build()

    def build(self):
        hxs = np.linspace(-self.L/2,self.L/2,7)
        xs = [hxs[_] for _ in [0,1,3,6,6,4,0,0]]
        ys = [hxs[_] for _ in [2,3,1,4,2,0,0,2]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [hxs[_] for _ in [2,3,4,2]]
        ys = [hxs[_] for _ in [6,5,6,6]]
        self.tiles.append(self._make_path(xs,ys))

class TRB2(TileBase):
    def __init__(self,L,x0,y0,flp=False):
        super().__init__(L,x0,y0,pf=0,flp=flp)
        self.build()

    def build(self):
        hxs = np.linspace(-self.L/2,self.L/2,7)
        xs = [self.xs[_] for _ in [0,1,0,0]]
        ys = [self.xs[_] for _ in [2,3,3,2]]
        self.tiles = [self._make_path(xs,ys)]
        xs = [hxs[_] for _ in [2,5,2,4,6,6,4,2]]
        ys = [hxs[_] for _ in [6,3,0,0,2,4,6,6]]
        self.tiles.append(self._make_path(xs,ys))

def tester(L=1):
    F = plt.figure(figsize=(7.75,4.5),frameon=False,dpi=184)
    ax = plt.Axes(F,[0,0,1.0,1.0])
    ax.set_aspect('auto')
    ax.set_axis_off()

    ax.fill_between([-1.5,6.25],3.5,-.5,color='seashell',alpha=1)

    tiles = []
    for i in range(3):
        for j in range(4):
            _ = np.random.choice([T1,T2,T3,T5,T4,T6,T7])
            tiles.append(_(L,(i-1)*L,j*L))
        

    for j in range(4):
        _ = np.random.choice((0,1))
        if _ == 0:
            _a = TRT1
            _b = TRB1
        else:
            _a = TRT2
            _b = TRB2
        tiles.append(_a(0.5,1.75,j*L+.25))
        tiles.append(_b(0.5,1.75,j*L-.25))
    for i in range(4):
        for j in range(8):
            _ = np.random.choice([T1,T2,T3,T5,T4,T6,T7])
            tiles.append(_(.5,2.25+i*.5,-.25+j*.5))

    for j in range(8):
        _ = np.random.choice((0,1))
        if _ == 0:
            _a = TRT1
            _b = TRB1
        else:
            _a = TRT2
            _b = TRB2
        tiles.append(_a(.25,4.125,j*.5+.125-.25))
        tiles.append(_b(.25,4.125,j*.5-.125-.25))
    for i in range(8):
        for j in range(16):
            _ = np.random.choice([T1,T2,T3,T5,T4,T6,T7])
            tiles.append(_(.25,4.375+i*.25,-.375+j*.25))

    

    for tile in tiles:
        tile.add_to_ax(ax,'g','None')
    F.add_axes(ax)
    ax.set_xlim(-1.5,6.25)
    ax.set_ylim(-.5,3.5)
    plt.savefig('multiscale_test.png')

tester()