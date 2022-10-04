import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import imageio
import bezier

import matplotlib as mpl

class TileBase():
    def __init__(self,L,x0,y0,ang):
        ext = L*.002
        self.h = L*np.sqrt(0.75)
        self.c = -L*np.sqrt(3)/3.
        self.centre = np.array([[0],[-L*np.sqrt(3)/3.]])
        self.ang=ang
        self.xs = np.array([0,-L/2-ext,L/2+ext,0])
        self.ys = np.array([0+ext,-self.h-ext,-self.h-ext,0+ext])
        self.rot = np.random.randint(0,2,1)*(2.*np.pi)
        self.rmat = np.array([
            [np.cos(ang),(-1.)*np.sin(ang)],
            [np.sin(ang),np.cos(ang)]
        ])
        tang = ang+self.rot
        self.tmat = np.array([
            [np.cos(tang),(-1.)*np.sin(tang)],
            [np.sin(tang),np.cos(tang)]
        ])

        self.centre = np.dot(self.centre.T,self.rmat).T.squeeze()
        self.origin = (x0,y0)
        self.tiles = []
        self.L = L
        self.lines = []

    def _make_path(self,xs,ys):

        vs = np.vstack((np.array(xs),np.array(ys)))
        vs = np.dot(vs.T,self.rmat).T.squeeze()

        vs[0]+=self.origin[0]
        vs[1]+=self.origin[1]
        verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]

        codes = [Path.MOVETO] + [Path.LINETO]*(len(xs)-2) + [Path.CLOSEPOLY]

        return Path(verts,codes)

class trA2_a(TileBase):
    def __init__(self,L,x0,y0,ang):
        super().__init__(L,x0,y0,ang)
        self.build()

    def build(self):
        tmc = -self.h/1.3
        tmh = -self.h/1.1
        for _ in np.linspace(0,2*np.pi,4)[:-1]:

            xs = [-self.L/5,-self.L/5,0,self.L/5,self.L/5,-self.L/5]
            ys = [-self.h,tmh,tmc,tmh,-self.h,-self.h]
            vs = np.vstack((xs,ys))
            vs[1] -= self.c
            tmrm = np.array([
                [np.cos(_),(-1.)*np.sin(_)],
                [np.sin(_),np.cos(_)]
            ])
            vs = np.dot(vs.T,tmrm).T.squeeze()
            vs = np.dot(vs.T,self.rmat).T.squeeze()
            vs[0] += self.centre[0]
            vs[1] += self.centre[1]

            vs[0] += self.origin[0]
            vs[1] += self.origin[1]
            self.lines.append(
                ([vs[0][_] for _ in [0,1,2,3,4]],[vs[1][_] for _ in [0,1,2,3,4 ]])
            )

            verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]
            codes = [Path.MOVETO]+[Path.LINETO]*(len(xs)-2)+[Path.CLOSEPOLY]
            self.tiles.append(Path(verts,codes))

        
      
class trA2_b(TileBase):
    def __init__(self,L,x0,y0,ang):
        super().__init__(L,x0,y0,ang)
        self.build()

    def build(self):
        self.tiles = []
        tmc = -self.h/1.8
        tmh = -self.h/1.8
        tmx = (3*self.L/10)*np.tan(np.pi/6.)
        for _ in np.linspace(0,2*np.pi,4)[:-1]:

            xs = [-self.L/5,-self.L/5,0,self.L/5,self.L/5,-self.L/5]
            ys = [-self.h,tmh,tmc,tmh,-self.h,-self.h]
            vs = np.vstack((xs,ys))
            vs[1] -= self.c
            tmrm = np.array([
                [np.cos(_),(-1.)*np.sin(_)],
                [np.sin(_),np.cos(_)]
            ])
            vs = np.dot(vs.T,tmrm).T.squeeze()
            vs = np.dot(vs.T,self.rmat).T.squeeze()
            vs[0] += self.centre[0]
            vs[1] += self.centre[1]

            vs[0] += self.origin[0]
            vs[1] += self.origin[1]
            verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]
            codes = [Path.MOVETO]+[Path.LINETO]*(len(xs)-2)+[Path.CLOSEPOLY]
            self.tiles.append(Path(verts,codes))

            tmrm = np.array([
                [np.cos(_),(-1.)*np.sin(_)],
                [np.sin(_),np.cos(_)]
            ])
            
            for _x in [self.L/5,-self.L/5]:
                xs = [_x,_x]
                ys = [-self.h,-self.h+tmx]  

                vs = np.vstack((xs,ys))
                vs[1] -= self.c
                tmrm = np.array([
                    [np.cos(_),(-1.)*np.sin(_)],
                    [np.sin(_),np.cos(_)]
                ])
                vs = np.dot(vs.T,tmrm).T.squeeze()
                vs = np.dot(vs.T,self.rmat).T.squeeze()
                vs[0] += self.centre[0]
                vs[1] += self.centre[1]

                vs[0] += self.origin[0]
                vs[1] += self.origin[1]
                self.lines.append((vs[0],vs[1]))

class trA2_c(TileBase):
    def __init__(self,L,x0,y0,ang):
        super().__init__(L,x0,y0,ang)
        self.ext = 0#np.random.choice([_ for _ in np.linspace(0,2*np.pi,4)[:-1]])
        self.build()

    def build(self):
        self.tiles = []
        tmc = -self.h/1.8
        tmh = -self.h/1.3
        tmx = (3*self.L/10)*np.tan(np.pi/6.)
        verts = []
        codes = []
        c = -1
        for _ in np.linspace(0,2*np.pi,4)[1:-1]:

            xs = [-self.L/5,-self.L/5,0,self.L/5,self.L/5,-self.L/5]
            ys = [-self.h,tmh,tmc,tmh,-self.h,-self.h]
            vs = np.vstack((xs,ys))
            vs[1] -= self.c
            tmrm = np.array([
                [np.cos(_+self.ext),(-1.)*np.sin(_+self.ext)],
                [np.sin(_+self.ext),np.cos(_+self.ext)]
            ])
            vs = np.dot(vs.T,tmrm).T.squeeze()
            vs = np.dot(vs.T,self.rmat).T.squeeze()
            vs[0] += self.centre[0]
            vs[1] += self.centre[1]

            vs[0] += self.origin[0]
            vs[1] += self.origin[1]
            verts += [(vs[0][i],vs[1][i]) for i in range(len(xs))]
            codes += [Path.MOVETO]+[Path.LINETO]*(len(xs)-2)+[Path.CLOSEPOLY]
            self.tiles.append(Path(verts,codes))  

            ptch = patches.PathPatch(self.tiles[-1])
            tmv = ptch.get_patch_transform().transform(vs)
            self.lines.append(tmv)

            for _x in [c*self.L/5]:
                xs = [_x,_x]
                ys = [-self.h,-self.h+tmx]  

                vs = np.vstack((xs,ys))
                vs[1] -= self.c
                tmrm = np.array([
                    [np.cos(_),(-1.)*np.sin(_)],
                    [np.sin(_),np.cos(_)]
                ])
                vs = np.dot(vs.T,tmrm).T.squeeze()
                vs = np.dot(vs.T,self.rmat).T.squeeze()
                vs[0] += self.centre[0]
                vs[1] += self.centre[1]

                vs[0] += self.origin[0]
                vs[1] += self.origin[1]
                self.lines.append((vs[0],vs[1])) 
            c+=2

        tmc = -self.h/1.1
        tmh = -self.h/1.0
        for _ in np.linspace(0,2*np.pi,4)[:-1]:

            xs = [-self.L/5,-self.L/5,0,self.L/5,self.L/5,-self.L/5]
            ys = [-self.h,tmh,tmc,tmh,-self.h,-self.h]
            vs = np.vstack((xs,ys))
            vs[1] -= self.c
            tmrm = np.array([
                [np.cos(_+self.ext),(-1.)*np.sin(_+self.ext)],
                [np.sin(_+self.ext),np.cos(_+self.ext)]
            ])
            vs = np.dot(vs.T,tmrm).T.squeeze()
            vs = np.dot(vs.T,self.rmat).T.squeeze()
            vs[0] += self.centre[0]
            vs[1] += self.centre[1]

            vs[0] += self.origin[0]
            vs[1] += self.origin[1]
            verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]
            codes = [Path.MOVETO]+[Path.LINETO]*(len(xs)-2)+[Path.CLOSEPOLY]
            self.tiles.append(Path(verts,codes))
            self.lines.append((
                [vs[0][_] for _ in [0,1,2,3,4]],
                [vs[1][_] for _ in [0,1,2,3,4]]
            ))
            break  

class trA2_1(TileBase):
    def __init__(self,L,x0,y0,ang):
        super().__init__(L,x0,y0,ang)
        self.build()

    def make_curve(self,N,v):
        x = [_[0] for _ in v]
        y = [_[1] for _ in v]
        n = np.vstack((x,y))
        curve = bezier.Curve(n,degree=3)
        out = np.zeros((2,N))
        for _,i in enumerate(np.linspace(0,1,N)):
            t = curve.evaluate(i).T[0]
            out[0,_] = t[0]
            out[1,_] = t[1]
        return out

    def build(self):
        self.tiles = []

        
        for _ in np.linspace(0,2*np.pi,4)[:-1]:
            tmh = -self.h/np.random.uniform(1.2,1.5)

            xs = [-self.L/5,-self.L/5,self.L/5,self.L/5,-self.L/5]
            ys = [-self.h,tmh,tmh,-self.h,-self.h]
            vs = np.vstack((xs,ys))
            vs[1] -= self.c
            tmrm = np.array([
                [np.cos(_),(-1.)*np.sin(_)],
                [np.sin(_),np.cos(_)]
            ])
            vs = np.dot(vs.T,tmrm).T.squeeze()
            vs = np.dot(vs.T,self.rmat).T.squeeze()
            vs[0] += self.centre[0]
            vs[1] += self.centre[1]

            vs[0] += self.origin[0]
            vs[1] += self.origin[1]
            verts = [(vs[0][i],vs[1][i]) for i in range(len(xs))]
            codes = [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY]
        
            self.tiles.append(Path(verts,codes))
            self.lines.append(self.make_curve(50,verts[:-1]))
            

class trA2_2(TileBase):
    def __init__(self,L,x0,y0,ang):
        super().__init__(L,x0,y0,ang)
        self.build()

    def make_curve(self,N,v):
        x = [_[0] for _ in v]
        y = [_[1] for _ in v]
        n = np.vstack((x,y))
        curve = bezier.Curve(n,degree=2)
        out = np.zeros((2,N))
        for _,i in enumerate(np.linspace(0,1,N)):
            t = curve.evaluate(i).T[0]
            out[0,_] = t[0]
            out[1,_] = t[1]
        return out

    def build(self):
        self.tiles = []

        verts = []
        codes = []
        for _ in np.linspace(0,-2*np.pi,4)[:-1]:
            tmh = -self.h/1.2

            x0= [-self.L/5,self.L/5,self.L/5.5]
            y0 = [-self.h,-self.h,tmh]
            vs = np.vstack((x0,y0))
            vs[1] -= self.c
            tmrm = np.array([
                [np.cos(_),(-1.)*np.sin(_)],
                [np.sin(_),np.cos(_)]
            ])
            vs = np.dot(vs.T,tmrm).T.squeeze()
            vs = np.dot(vs.T,self.rmat).T.squeeze()
            vs[0] += self.centre[0]
            vs[1] += self.centre[1]

            #vs = np.dot(vs.T,self.rmat).T.squeeze()

            vs[0] += self.origin[0]
            vs[1] += self.origin[1]
            verts += [(vs[0][i],vs[1][i]) for i in range(len(x0))]
            if _ == 0:
                codes += [Path.MOVETO,Path.LINETO,Path.CURVE3]
            else:
                codes += [Path.CURVE3,Path.LINETO,Path.CURVE3]
        
        verts.append(verts[0])
        codes.append(Path.CURVE3)

        self.tiles.append(Path(verts,codes))
        self.vs = verts

        i = 1
        for _ in range(3):
            n = []
            for x in range(3):
                n.append(verts[i])
                i+=1
            self.lines.append(self.make_curve(50,n))

L = 1
h = L*np.sqrt(0.75)

F = plt.figure(figsize=(7.5,11.2),frameon=False,dpi=100)  
ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
ax.set_aspect('equal')
ax.set_axis_off()
F.add_axes(ax)

L2 = 3
h2 = L2*np.sqrt(0.75)
for _x in range(5):
    for _y in range(6):
        x0,y0 = _x*1.5*L2,_y*2*h2
        if _x%2 == 0: y0-=h2
        for a in np.linspace(0,2*np.pi,7)[:-1]:
            x = np.random.choice([trA2_c,trA2_a,trA2_1,trA2_2,trA2_b])
            t = x(L2,x0,y0,a)
            m = t._make_path(t.xs,t.ys)
            #p = patches.PathPatch(m,fc='gold',ec='None')
            #ax.add_patch(p)
            for m in t.tiles:
                p = patches.PathPatch(m,fc='tab:orange',ec='None',zorder=400)
                ax.add_patch(p)
            for l in t.lines:
                ax.plot(l[0],l[1],'-',c='darkgoldenrod',lw=3)

for _x in range(6):
    for _y in range(7):
        x0,y0 = _x*1.5*L,_y*2*h
        if _x%2 == 0: y0-=h
        for a in np.linspace(0,2*np.pi,7)[:-1]:
            x = np.random.choice([trA2_c,trA2_a,trA2_1,trA2_2,trA2_b])
            t = x(L,x0,y0,a)
            m = t._make_path(t.xs,t.ys)
            p = patches.PathPatch(m,fc='gold',ec='None',alpha=.7,zorder=600)
            ax.add_patch(p)
            for m in t.tiles:
                p = patches.PathPatch(m,fc='dodgerblue',ec='None',zorder=1000)
                ax.add_patch(p)
            for l in t.lines:
                ax.plot(l[0],l[1],'-',c='darkblue',lw=3,zorder=995)

ax.set_xlim(0,7.5)
ax.set_ylim(-0.9,10.3)
plt.savefig('tra2.png')