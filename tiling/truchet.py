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
    def __init__(self,L,x0,y0,pf=2,flp=False):
        self.xs = np.linspace(-L/2,L/2,4)
        self.rot = np.random.randint(0,3,1)*np.pi/pf if pf != 0 else 0
        if flp:
            self.rot -= np.pi/2
        self.rmat = np.array([
            [np.cos(self.rot),(-1.)*np.sin(self.rot)],
            [np.sin(self.rot),np.cos(self.rot)]
        ])
        self.origin = (x0,y0)
        self.tiles = []
        self.L = L
        self.lines = []
        self.flp = flp

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
    
    def add_to_ax(self,ax,fc,lc,zorder=1,lw=1):
        for tile in self.tiles:
            p = patches.PathPatch(tile,fc=fc,ec='None',zorder=zorder)
            ax.add_patch(p)
        if lc != 'None':
            for line in self.lines:
                ax.plot(*line,'-',c=lc,lw=lw,zorder=zorder)

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
    def __init__(self,L,x0,y0,pf=2,flp=False):
        super().__init__(L,x0,y0,pf=pf,flp=flp)
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
    def __init__(self,L,x0,y0,pf=2,flp=False):
        super().__init__(L,x0,y0,pf=pf,flp=flp)
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
    def __init__(self,L,x0,y0,pf=2,flp=False):
        super().__init__(L,x0,y0,pf=pf,flp=flp)
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
    def __init__(self,L,x0,y0,pf=2,flp=False):
        super().__init__(L,x0,y0,pf=pf,flp=flp)
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
    def __init__(self,L,x0,y0,pf=2,flp=False):
        super().__init__(L,x0,y0,pf=pf,flp=flp)
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

class R1(TileBase):
    def __init__(self,L,x0,y0,w,flp=False,cflip=False,pf=2):
        super().__init__(L,x0,y0,pf=pf,flp=flp)
        self.w = (self.L/2)*w
        self.s = self.xs[1]+self.w/2
        self.e = self.xs[1]-self.w/2
        self.flp = flp
        self.cflip = cflip
        self.build()

    def _make_path(self,xs,ys,codes=[]):

        vs = np.vstack((np.array(xs),np.array(ys)))
        vs = np.dot(vs.T,self.rmat).T.squeeze()

        vs[0]+=self.origin[0]
        vs[1]+=self.origin[1]
        verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]
        if len(codes) == 0:
            codes = [Path.MOVETO] + [Path.LINETO]*(len(xs)-2) + [Path.CLOSEPOLY]
        return Path(verts,codes)
    
    def build(self):
        dx,dy = np.random.uniform(-self.L/2,0),np.random.uniform(0,self.L/2)
        self.tiles = []
        self.lines = []
        xs = [
            self.xs[3],self.e+dx,self.e,self.e,self.s,self.s,self.s+dx,self.xs[3],self.xs[3]
        ]
        ys = [
            -self.e,-self.e,-self.e-dy,self.xs[0],self.xs[0],-self.s-dy,-self.s,-self.s,-self.e
        ]


        codes = [Path.MOVETO] + [Path.CURVE4]*3 + [Path.LINETO] + [Path.CURVE4]*3 + [Path.CLOSEPOLY]

        self.tiles = [self._make_path(xs,ys,codes=codes)]

        dx,dy = np.random.uniform(-self.L/2,0),np.random.uniform(0,self.L/2)
        xs = [
            self.xs[0],-self.e-dx,-self.e,-self.e,-self.s,-self.s,-self.s-dx,self.xs[0],self.xs[0]
        ]
        ys = [
            self.e,self.e,self.e+dy,self.xs[3],self.xs[3],self.s+dy,self.s,self.s,self.e
        ]
        

        self.tiles.append(self._make_path(xs,ys,codes=codes))

def make_new_column(Y,i,L=1):
    return [np.random.choice([T1,T2,T3,T4,T4,T4,T5,T5,T5,T6,T7])(L,i*L,j*L) for j in range(Y)]

def make_figure(tiles,tkeys,X,Y,x0,y0,L=1,c1='g',c2='gold',mf = 9):
    if X > Y:
        fx = mf
        fy = fx*(Y/X)
    else:
        fy = mf
        fx = fy*(X/Y)
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
    

def output0(c1 = 'maroon',c2 = 'tab:orange',c3 = 'darkgoldenrod',c4 = 'magenta',c5 = 'darkkhaki',c6 = 'beige', c7 = 'w',fname='tst.png'):
    L = 1

    X,Y = 25,31
    mf = 9
    if X > Y:
        fx = mf
        fy = fx*(Y/X)
    else:
        fy = mf
        fx = fy*(X/Y)

    tiles = []
    for i in range(X):
        for j in range(Y):
            _ = np.random.choice([T4,T4,T4,T4,T4,T4,T4,T5,T5,T6,T7])
            tt = _(L,i*L,j*L,pf=2)
            tiles.append(tt)

    x = Squartet(3*X,3*Y,0.4,yi=1,xi=0,maxang=np.pi/2.2,rfrc=0.5)
    omega = np.random.uniform(0,2*np.pi,1)
    x.spin(omega)
    x.make_patches()

    F = plt.figure(figsize=(fx,fy),frameon=False)
    ax = plt.Axes(F,[0,0,1.0,1.0])
    ax.set_aspect('auto')
    ax.set_axis_off()
    ax.fill_between([-L/2,L*X-L/2],L*Y-L/2,-L/2,color='seashell',alpha=1)
    ax.fill_between([-L/2,L*X-L/2],L*Y-L/2,-L/2,color=c5,alpha=.3)
    F.add_axes(ax)
    for tile in x.paths:
        v = np.array(tile.vertices)/3.5 - L/2
        a = 0.3*np.mean(v.T[1])/Y+0.05
        v = [list(_v) for _v in v]
        _tile = Path(v,tile.codes)
        p = patches.PathPatch(_tile,fc=c3,ec='None',zorder=5,alpha=a)
        ax.add_patch(p)
    for j,tile in enumerate(tiles):
        a = tile.origin[1]/Y
        tile.add_to_ax(ax,colorFader(c1,c2,a),colorFader(c1,c4,a),zorder=10,lw=3.5)
    ax.fill_between([-L/2,0],L*Y-L/2,-L/2,color='seashell',alpha=1,zorder=15)
    ax.fill_between([-L/2,L*X-L/2],L*Y-L/2,L*Y-L,color='seashell',alpha=1,zorder=15)
    ax.fill_between([-L/2,L*X-L/2],0,-L/2,color='seashell',alpha=1,zorder=15)
    ax.fill_between([L*X-L/2,L*X-L],L*Y-L/2,-L/2,color='seashell',alpha=1,zorder=15)
    ax.plot([0,L*X-L,L*X-L,0,0],[0,0,L*Y-L,L*Y-L,0],'-',c=c6,lw=4,zorder=20)
    ax.plot([0,L*X-L,L*X-L,0,0],[0,0,L*Y-L,L*Y-L,0],'-',c=c7,lw=1,zorder=22)
    ax.set_xlim(-L/2,L*X-L/2)
    ax.set_ylim(-L/2,L*Y-L/2)
    plt.savefig(fname)
    plt.close()

def output1():

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

class Squartle():

    def __init__(self,L,c,max_a):
        self.L = L
        self.c = c
        self.m = max_a
        self.paths = []
        self.inner = np.array([
            [0,L],[L,0],[0,-L],[-L,0]
        ])
        self.outer = np.array([
            np.vstack(([-1+L+c[0],1-L+c[0]],[1+c[1],1+c[1]])),
            np.vstack(([1+c[0],1+c[0]],[1-L+c[1],-1+L+c[1]])),
            np.vstack(([-1+L+c[0],1-L+c[0]],[-1+c[1],-1+c[1]])),
            np.vstack(([-1+c[0],-1+c[0]],[1-L+c[1],-1+L+c[1]])),
        ])
        
    def make_path(self,r,af):
        a = af*self.m
        rmat = np.array([
            [np.cos(a),(-1.)*np.sin(a)],
            [np.sin(a),np.cos(a)]
        ])
        self.rmat = rmat
        
        inner = np.dot(self.inner,rmat).T.squeeze()
        inner[0] += self.c[0]
        inner[1] += self.c[1]


        self.paths = []
        for i,o in zip(inner.T,self.outer):

            verts = [(o[0][j],o[1][j]) for j in range(len(o))]
            verts += [(i[0],i[1])]
            verts += [verts[0]]
            codes = [Path.MOVETO] + [Path.LINETO]*2 + [Path.CLOSEPOLY]
            self.paths.append(Path(verts,codes))

    def add_to_ax(self,ax,fc,lc=None):
        for tile in self.paths:
            p = patches.PathPatch(tile,fc=fc,ec='None')
            ax.add_patch(p)
        # for line in self.lines:
        #     ax.plot(*line,'-',c=lc,lw=1)

def make_figure2(tiles,X,Y,L=1,c1='g',c2='gold',mf = 5):
    if X > Y:
        fx = mf
        fy = fx*(Y/X)
    else:
        fy = mf
        fx = fy*(X/Y)
    F = plt.figure(figsize=(fx,fy),frameon=False,dpi=250)
    ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
    ax.set_aspect('auto')
    ax.set_axis_off()
    #ax.fill_between([L/2,L*X-L/2],L*Y-L/2,L/2,color='w')
    F.add_axes(ax)
    for tile in tiles:
        p = patches.PathPatch(tile,fc=c1,ec='None')
        ax.add_patch(p)
    ax.set_xlim(-L,2*X-L)
    ax.set_ylim(-L,2*Y-L)
    plt.savefig('frame.png')
    plt.close()

def make_figure3(r1s,X,Y,c1='tab:red',c2='tab:blue',mf=3):
    if X > Y:
        fx = mf
        fy = fx*(Y/X)
    else:
        fy = mf
        fx = fy*(X/Y)

    

    F = plt.figure(figsize=(fx,fy),frameon=False,dpi=250)
    ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
    ax.set_aspect('auto')
    ax.set_axis_off()
    F.add_axes(ax)
    mg = []
    for r1 in r1s:
        c = [c1,c2] if r1.flp+r1.cflip == 1 else [c2,c1]
        for i,tile in enumerate(r1.tiles):
            p = patches.PathPatch(tile,fc=c[i],ec='None',lw=.2)
            ax.add_patch(p)
        if len(mg) == 0:
            mgx,mgy = np.meshgrid(r1.xs,r1.xs)
        ax.plot(mgx.ravel()+r1.origin[0],mgy.ravel()+r1.origin[1],'o',alpha=0)
    
    ax.set_xlim(-.5,X-.5)
    ax.set_ylim(-.5,Y-.5)
    plt.savefig('frame.png')
    plt.close()
    

def output2():

    X,Y = 16,8
    images = []
    for omega in np.linspace(0,2*np.pi,50):
        tiles = []
        for i in range(X):
            omx = 2*i*np.pi/X+omega
            for j in range(Y):
                omy = 5*j*np.pi/Y+omega
                tt = Squartle(0.2,(2*i,2*j),np.pi/2.5)
                a = (np.sin(omx)+np.cos(omy))/2
                a = np.sin(omx)
                tt.make_path(9,a)
                tiles += tt.paths

        make_figure2(tiles,X,Y)
        images.append(imageio.imread('frame.png'))
    imageio.mimsave('./truch2.mp4',images,fps=15)
    
def output3():

    X,Y = 40,30
    x = Squartet(X,Y,.4,yi=3,maxang=np.pi/2.6)
    images = []
    for omega in np.linspace(0,2*np.pi,25):
        x.spin(omega)
        x.make_patches()
        x.add_to_ax()
        images.append(imageio.imread('frame.png'))
    images = images[:-1]
    imageio.mimsave('./truch3.mp4',images,fps=15)

def output4():
    X,Y = 15,10
    images = []
    ws = (0.8*(np.sin(np.linspace(0,4*np.pi,60))+1))/2. + 0.2
    for w in ws:
        tiles = []
        for i in range(X):
            for j in range(Y):
                flp = False
                cflip = False
                if i%2 == 0 and j%2 != 0:
                    flp = True
                if i%2 != 0 and j%2 == 0:
                    flp = True
                if i%2 == 0:
                    cflip = True
                tiles += [R1(1,i,j,w,flp=flp,cflip=cflip,pf=0)]
        make_figure3(tiles,X,Y)
        images.append(imageio.imread('frame.png'))
    imageio.mimsave('./truch4.mp4',images,fps=15)

class Squartet():
    def __init__(self,X,Y,L,yi=4,xi=2,maxang=np.pi/2,rfrc=.5):
        self.grid = np.meshgrid(
            (np.linspace(1,X+2,X+2),np.linspace(1,Y+2,Y+2))
        )
        self.X = X
        self.Y = Y
        self.m = maxang
        self.yi = yi
        self.xi = xi
        self.rfrc = rfrc

        self.base_points = np.array([
            [0,L],[L,0],[0,-L],[-L,0]
        ])

    def spin(self,w=0):
        self.ps = {}
        self.cs = {}
        for i in range(self.X):
            omx = (self.xi*i*np.pi/self.X) + w
            omx2 = 2*(self.yi*i*np.pi/self.X) + w
            for j in range(self.Y):
                omy = (self.yi*j*np.pi/self.Y) + w
                omy2 = (self.xi*i*np.pi/self.Y) + w
                af = (np.sin(omx)+np.sin(omy))/2.
                af2 = (np.sin(omx2)+np.sin(omy2))/2.
                if i%2 == 0: 
                    af2 *= -1
                if j%2 == 0: af2 *= -1
                a = af*self.m - np.pi/4.
                rmat = np.array([
                    [np.cos(a),(-1.)*np.sin(a)],
                    [np.sin(a),np.cos(a)]
                ])
                ps = self.base_points.copy()*(1-self.rfrc*af2)
                ps = np.dot(ps,rmat).T.squeeze()
                ps[0] += 2*i
                ps[1] += 2*j
                self.ps[f'{i},{j}'] = ps
                cf = af
                if cf < 0: cf = 0
                if cf > 1: cf = 1
                self.cs[f'{i},{j}'] = colorFader('slategrey','pink',mix=cf)


    def make_patches(self):
        self.paths = []
        self.tc = []
        for i in range(1,self.X):
            for j in range(1,self.Y):
                verts = [
                    tuple(self.ps[f'{i-1},{j-1}'].T[1]),
                    tuple(self.ps[f'{i-1},{j}'].T[2]),
                    tuple(self.ps[f'{i},{j}'].T[3]),
                    tuple(self.ps[f'{i},{j-1}'].T[0]),
                ]
                verts += [verts[0]]
                codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
                self.paths.append(Path(verts,codes))
                self.tc.append(self.cs[f'{i},{j}'])

    def add_to_ax(self):
        mf = 9
        if self.X > self.Y:
            fx = mf
            fy = fx*(self.Y/self.X)
        else:
            fy = mf
            fx = fy*(self.X/self.Y)
        F = plt.figure(figsize=(fx,fy),frameon=False,dpi=250)
        ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
        ax.set_aspect('auto')
        ax.set_axis_off()
        #ax.fill_between([L/2,L*X-L/2],L*Y-L/2,L/2,color='w')
        F.add_axes(ax)
        for tile,fc in zip(self.paths,self.tc):
            p = patches.PathPatch(tile,fc=fc,ec='k')
            ax.add_patch(p)
        # for k,i in self.ps.items():
        #     ax.plot(i[0],i[1],'ko')
        ax.set_xlim(-1,2*self.X-1)
        ax.set_ylim(-1,2*self.Y-1)
        plt.savefig('frame.png')
        plt.close()

def outputA(c1 = 'maroon',c2 = 'tab:orange',c3 = 'darkgoldenrod',c4 = 'magenta',c5 = 'darkkhaki',c6 = 'beige', c7 = 'w',c8='pink',c9='crimson',fname='tst.png'):
    L = 1

    X,Y = 15,20
    mf = 9
    if X > Y:
        fx = mf
        fy = fx*(Y/X)
    else:
        fy = mf
        fx = fy*(X/Y)

    tiles = []
    for i in range(X):
        for j in range(Y):
            _ = np.random.choice([T4,T4,T4,T6,T7])#T4,T5,T5,T6,T7])
            for flp in [True,False]:
                dx = L/2 if flp else 0
                tt = _(L,i*L+dx,j*L,pf=0,flp=flp)
                tiles.append(tt)

    x = Squartet(3*X,3*Y,0.4,yi=0,xi=2,maxang=np.pi/2.,rfrc=0.2)
    omega = np.random.uniform(0,2*np.pi,1)
    x.spin(omega)
    x.make_patches()

    F = plt.figure(figsize=(fx,fy),frameon=False)
    ax = plt.Axes(F,[0,0,1.0,1.0])
    ax.set_aspect('auto')
    ax.set_axis_off()
    ax.fill_between([-L/2,L*X-L/2],L*Y-L/2,-L/2,color='seashell',alpha=1)
    ax.fill_between([-L/2,L*X-L/2],L*Y-L/2,-L/2,color=c5,alpha=.3)
    F.add_axes(ax)
    for tile in x.paths:
        v = np.array(tile.vertices)/3.5 - L/2
        a = 0.3*np.mean(v.T[1])/Y+0.05
        v = [list(_v) for _v in v]
        _tile = Path(v,tile.codes)
        p = patches.PathPatch(_tile,fc=c3,ec='None',zorder=5,alpha=a)
        ax.add_patch(p)

    zo = np.linspace(1,len(tiles),len(tiles)) + 10
    np.random.shuffle(zo)
    for j,tile in enumerate(tiles):
        a = tile.origin[1]/Y
        _zo = zo[j]
        _c1,_c2 = [colorFader(c1,c2,a),colorFader(c1,c4,a)] if tile.flp else [colorFader(c5,c8,a),colorFader(c5,c9,a)]
        tile.add_to_ax(ax,_c1,_c2,zorder=_zo,lw=1.5)
    ax.fill_between([-L/2,0],L*Y-L/2,-L/2,color='seashell',alpha=1,zorder=15+zo.max())
    ax.fill_between([-L/2,L*X-L/2],L*Y-L/2,L*Y-L,color='seashell',alpha=1,zorder=15+zo.max())
    ax.fill_between([-L/2,L*X-L/2],0,-L/2,color='seashell',alpha=1,zorder=15+zo.max())
    ax.fill_between([L*X-L/2,L*X-L],L*Y-L/2,-L/2,color='seashell',alpha=1,zorder=15+zo.max())
    ax.plot([0,L*X-L,L*X-L,0,0],[0,0,L*Y-L,L*Y-L,0],'-',c=c6,lw=4,zorder=20+zo.max())
    ax.plot([0,L*X-L,L*X-L,0,0],[0,0,L*Y-L,L*Y-L,0],'-',c=c7,lw=1,zorder=22+zo.max())
    ax.set_xlim(-L/2,L*X-L/2)
    ax.set_ylim(-L/2,L*Y-L/2)
    plt.savefig(fname)
    plt.close()