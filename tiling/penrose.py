import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

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
        self.centre = np.array(self.verts).mean(axis=0)

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
        self._make_arcs()

    def _make_arcs(self):
        al1 = self.phi-self.L
        al2 = 2*al1/(1.+np.sqrt(5))

        sa,ea = -self.angle-self.a1,-self.angle+self.a1
        theta = np.linspace(sa,ea,100)
        self.arc1 = np.vstack( (
            al1*np.cos(theta)+self.verts[0][0],
            al1*np.sin(theta)+self.verts[0][1]
        ))

        sa,ea = -self.angle-2*self.a1,-self.angle+2*self.a1
        theta = np.linspace(sa,ea,100)+np.pi
        self.arc2 = np.vstack((
            al2*np.cos(theta)+self.verts[2][0],
            al2*np.sin(theta)+self.verts[2][1]
        ))

    def inflate(self,cents):
        l = self.phi - self.L
        k1 = Kite(self.verts[1],np.pi+self.angle-2*self.a1,l)
        k2 = Kite(self.verts[3],np.pi+self.angle+2*self.a1,l)
        d1 = Dart(self.verts[0],np.pi+self.angle+4*self.a1,l)
        d2 = Dart(self.verts[0],np.pi+self.angle-4*self.a1,l)

        ts = [k1,k2,d1,d2]
        return [t for t in ts if np.sqrt(((cents-t.centre)*(cents-t.centre)).sum(axis=1)).min() > self.L*0.0001]

class Dart(P2_base):

    def __init__(self,point,angle,l):
        super().__init__(point,angle,l)
        self.type='Dart'
        self.xs = [0,l*np.cos(self.a1),l*l/self.phi,l*np.cos(self.a1),0]
        self.ys = [0,l*np.sin(self.a1),0,-l*np.sin(self.a1),0]
        self._make_path()
        self._make_arcs()

    def _make_arcs(self):
        dx,dy = self.verts[1][0]-self.verts[2][0],self.verts[1][1]-self.verts[2][1]
        pt = np.sqrt((dx*dx)+(dy*dy))
        al1 = 2*pt/(1.+np.sqrt(5))
        al2 = pt-al1

        sa,ea = -self.angle-self.a1,-self.angle+self.a1
        theta = np.linspace(sa,ea,100)
        self.arc1 = np.vstack((
            al1*np.cos(theta)+self.verts[0][0],
            al1*np.sin(theta)+self.verts[0][1]
        ))

        sa,ea = -self.angle-3*self.a1,-self.angle+3*self.a1
        theta = np.linspace(sa,ea,100)+np.pi
        self.arc2 = np.vstack((
            al2*np.cos(theta)+self.verts[2][0],
            al2*np.sin(theta)+self.verts[2][1]
        ))

    def inflate(self,cents):
        l = self.phi - self.L
        k = Kite(self.verts[0],self.angle,l)
        d1 = Dart(self.verts[1],np.pi+self.angle-self.a1,l)
        d2 = Dart(self.verts[3],np.pi+self.angle+self.a1,l)
        ts = [k,d1,d2]
        return [t for t in ts if np.sqrt(((cents-t.centre)*(cents-t.centre)).sum(axis=1)).min() > self.L*0.001]

class P3_base():

    def __init__(self,point,angle,l,a1,flip=False):
        self.L = l
        self.angle = angle
        self.point = point
        self.xs = [0,l*np.cos(a1/2.),2*l*np.cos(a1/2.),l*np.cos(a1/2),0]
        self.ys = [0,l*np.sin(a1/2.),0,-l*np.sin(a1/2.),0]
        if flip:
            self.xs = [self.xs[_] for _ in [2,3,0,1,2]]
            self.ys = [self.ys[_] for _ in [2,1,0,3,2]]
            self.xs = [self.xs[_]-self.xs[0] for _ in [0,1,2,3,0]]
            self.angle += np.pi
        self._make_path()

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
        self.centre = np.array(self.verts).mean(axis=0)

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self.path = Path(self.verts,codes)

class P3_skinny(P3_base):

    def __init__(self,point,angle,l,flip=False):
        self.type='P3skinny'
        a1 = 36.*np.pi/180.
        super().__init__(point,angle,l,a1,flip=flip)

    def inflate(self):
        a2 = 72.*np.pi/180.
        a3 = 18.*np.pi/180.
        l = 2*self.L*np.cos(72*np.pi/180)
        s1 = P3_skinny(self.verts[1],self.angle+a2,l,flip=True)
        s2 = P3_skinny(self.verts[1],self.angle+1.5*a2,l)
        f1 = P3_fat(self.verts[1],self.angle-a3+np.pi,l)
        f2 = P3_fat(self.verts[1],self.angle+a3,l)

        return s1,s2,f1,f2

class P3_fat(P3_base):

    def __init__(self,point,angle,l):
        self.type='P3fat'
        a1 = 72.*np.pi/180.
        super().__init__(point,angle,l,a1)
        
    def inflate(self):
        a2 = 36.*np.pi/180.
        l = 2*self.L*np.cos(72*np.pi/180)

        f1 = P3_fat(self.verts[2],self.angle+a2+np.pi,l)
        f2 = P3_fat(self.verts[2],self.angle-a2+np.pi,l)
        f3 = P3_fat(f1.verts[1],self.angle+np.pi,l)
        s1 = P3_skinny(f1.verts[1],self.angle+3.5*a2,l)
        s2 = P3_skinny(f1.verts[1],self.angle-3.5*a2,l,flip=True)

        return f1,f2,f3,s1,s2

if __name__ == '__main__':
    """
    F = plt.figure(frameon=False,figsize=(8,12.8),dpi=250)
    ax = plt.Axes(F,[0.,0.,1.,1.])
    ax = F.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    tiles = [Dart([0,0],0,1)]
    cents = np.array([tiles[0].centre])
    for i in range(8):
        tmt = []
        for t in tiles:
            tmt+=t.inflate(cents)
            cents = np.array([t.centre for t in tmt])
        tiles = tmt
    
    for t in tiles:
        c1 = 'tab:orange' if t.type == 'Kite' else 'tab:blue'
        c2 = 'gold' if t.type == 'Kite' else 'tab:purple'
        if t.centre[1] <= -0.32:
            c = c1
        elif t.centre[1] >= 0.32:
            c = c2
        else:
            mix = (t.centre[1]+.32)/.64
            c = colorFader(c1,c2,mix)
        p = patches.PathPatch(t.path,fc=c,ec='None')
        ax.add_patch(p)
        #ax.plot(t.arc1[0],t.arc1[1],'w-')
        ax.plot(t.arc2[0],t.arc2[1],'w-',lw=2)
    ax.set_xlim(.33,.71)
    ax.set_ylim(-.32,.32)
    plt.savefig('tmp.png')
    """

    A = P3_fat([0,0],-np.pi/2,1)
    tiles = A.inflate()
    F = plt.figure()
    ax = F.add_subplot(111)
    p = patches.PathPatch(A.path)
    ax.add_patch(p)
    for t in tiles:
        p = patches.PathPatch(t.path,fc='tab:blue')
        ax.add_patch(p)

    #cs = ['r','g','b','m','gold']
    #for t,c in zip(tiles[::-1],cs):
    #    #if t.type == 'P3skinny': continue
    #    for T in t.inflate():
    #        p = patches.PathPatch(T.path,fc=c,edgecolor=c)
    #        ax.add_patch(p)
    
    for i in range(5):
        tmt = []
        for t in tiles:
            tmt+=t.inflate()
        tiles = tmt

    for t in tiles:
        c = 'grey' if t.type == 'P3skinny' else 'tab:blue'
        p = patches.PathPatch(t.path,fc=c)
        ax.add_patch(p)

    ax.set_ylim(-.2,2.)
    ax.set_xlim(-1.2,1.2)
    ax.set_aspect('equal')
    plt.show()