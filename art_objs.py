import numpy as np
import matplotlib.pyplot as plt

def rotate_coords(crds,ang):
    rmat = np.array([[np.cos(ang),(-1.)*np.sin(ang)],
                         [np.sin(ang),np.cos(ang)]])

    return np.dot(crds.T,rmat).T

class SimpleLine(object):

    def __init__(self,x1,x2,y1,y2):
        self.m = (y2-y1)/(x2-x1)
        self.b = y1-(self.m*x1)

    def get_y(self,x):
        return (self.m*x)+self.b

class SimpleCircle(object):

    def __init__(self,xc=0,yc=0,radius=1.,npoints=1000):

        self.r = radius
        self.cent = np.array([xc,yc])

        self.ang = np.linspace(0.,np.pi*2.,npoints)
        self.x = xc+(self.r*np.cos(self.ang))
        self.y = yc+(self.r*np.sin(self.ang))

class SimpleEllipse(object):

    def __init__(self,a,r=1.,npoints=1000):

        self.a=a
        self.x = np.linspace(-1,1,int(npoints/2.))
        self.y = np.sqrt((a*a)*(1.-(self.x*self.x)))
        self.x = np.concatenate((self.x,np.flip(self.x)))
        self.y = np.concatenate((self.y,(-1.)*self.y))

class SinuCirc(object):

    def __init__(self,radius,inner_omega):
        self.r = radius
        self.om = inner_omega
        
        self.x = np.linspace((-1.)*radius,radius,5000)
        self.t = np.arccos(self.x/radius)
        self.yi = np.sin(self.t)*radius
        self.yo = np.sin(self.t*inner_omega)*radius
        self.y = self.yi*self.yo/radius
        self.x1 = np.copy(self.x)
        self.y1 = np.copy(self.yi)
        self.y2 = np.copy(self.yo)

    def Spin(self,theta):

        rmat = np.array([[np.cos(theta),(-1.)*np.sin(theta)],
                               [np.sin(theta),np.cos(theta)]])
        xy   = np.vstack((self.x,self.y))
        self.x = np.sum(xy.T*rmat[0],axis=1)
        self.y = np.sum(xy.T*rmat[1],axis=1)
          
class PowerFlower(object):

    def __init__(self,N=6,xs=0.,ys=0.,npoints=1000,rad=1.):

        self.xs = xs
        self.ys = ys
        self.N = N
        self.npoints = npoints
        self.rad=rad
        
        cent = SimpleCircle(xc=xs,yc=ys,npoints=self.npoints,radius=self.rad)
        self.angs = np.linspace(0,np.pi*2.,N+1)[:-1]
        self.xc,self.yc = np.cos(self.angs),np.sin(self.angs)
        self.layer = 0
        self.values = {f'layer{self.layer}':{'coords':np.vstack((rad*cent.x,rad*cent.y)),
                                             'cents':np.array([self.xc,self.yc])}
                      }

    def add_layer(self):
        
        self.layer+=1
        sc = self.rad*self.layer
        tx,ty = [],[]
        for i,ang in enumerate(self.angs):
            tx.append((self.rad*self.layer*self.xc[i])+self.xs)
            ty.append((self.rad*self.layer*self.yc[i])+self.ys)

            if self.layer > 1:
                try:
                    x1,x2 = (sc*self.xc[i])+self.xs,(sc*self.xc[i+1])+self.xs
                    y1,y2 = (sc*self.yc[i])+self.ys,(sc*self.yc[i+1])+self.ys
                    
                except:
                    x1,x2 = (sc*self.xc[i])+self.xs,(sc*self.xc[0])+self.xs
                    y1,y2 = (sc*self.yc[i])+self.ys,(sc*self.yc[0])+self.ys
                    
                if np.abs(x2-x1) >= 1.e-6:
                    tline = SimpleLine(x1,x2,y1,y2)
                    xs = np.linspace(x1,x2,self.layer+1)[1:-1]
                    ys = tline.get_y(xs)
                else:
                    xs = []
                    for i in range(self.layer-1): xs.append(x1)
                    ys = np.linspace(y2,y1,self.layer+1)[1:-1]
                for j in range(len(xs)):
                    tx.append(xs[j])
                    ty.append(ys[j])
        tmc = np.zeros((len(tx),2,self.npoints))
        for i in range(len(tx)):
            tcirc = SimpleCircle(xc=tx[i],yc=ty[i],npoints=self.npoints,radius=self.rad)
            tmc[i] = np.vstack((tcirc.x,tcirc.y))

            

        self.values[f'layer{self.layer}'] = {'coords':tmc}
        tx.append(tx[0])
        ty.append(ty[0])
        self.values[f'layer{self.layer}']['cents'] = np.vstack((tx,ty))
                    

class PowerElower(object):

    def __init__(self,N=6,layers=1,xs=0.,ys=0.,npoints=1000):

        self.xs = xs
        self.ys = ys
        self.N = N
        self.npoints = npoints
        
        cent = SimpleCircle(npoints=self.npoints)
        self.angs = np.linspace(0,np.pi*2.,N+1)[:-1]
        self.xc,self.yc = np.cos(self.angs),np.sin(self.angs)
        self.layer = 0
        self.values = {f'layer{self.layer}':{'coords':np.vstack((cent.x,cent.y)),
                                             'cents':np.array([self.xc,self.yc])}
                      }

    def add_layer(self):
        
        self.layer+=1
        s,c = np.sin(self.angs[1]),np.cos(self.angs[1])
        ellip = s/np.sqrt(1.-((1.-c)*(1.-c)))
        tmell = SimpleEllipse(ellip)
        tmell.x += float(self.layer)
        
        tx,ty = [],[]
        for i,ang in enumerate(self.angs):
            tx.append((self.layer*self.xc[i])+self.xs)
            ty.append((self.layer*self.yc[i])+self.ys)


        tmc = np.zeros((len(tx),2,self.npoints))
        for i,ang in enumerate(self.angs):
            s,c = np.sin(ang),np.cos(ang)
            xt = (tmell.x*c)-(tmell.y*s)
            yt = (tmell.x*s)+(tmell.y*c)
            tmc[i] = np.vstack((xt,yt))

        self.values[f'layer{self.layer}'] = {'coords':tmc}
        tx.append(tx[0])
        ty.append(ty[0])
        self.values[f'layer{self.layer}']['cents'] = np.vstack((tx,ty))

# WORK N PROGRESS
class SpinZoomer():

    def __init__(self,N=5,ang=15.*(np.pi/180.)):
        self.sides = 5
        self.bangs = np.linspace(0,np.pi*2.,N+1)

        self.x = np.cos(self.bangs)
        self.y = np.sin(self.bangs)
        self.n = 0.

    def add_one(self,frac=.1):

        self.n += 1.
        angp  = frac*(self.bangs[1]-self.bangs[0])*self.n
        tmang = self.bangs+angp
        if self.n == 1:
            x1,x2 = self.x[0],self.x[1]
            y1,y2 = self.y[0],self.y[1]
        else:
            tx,ty = self.x[int(self.n)-1],self.y[int(self.n)-1]
            x1,x2 = tx[0],tx[1]
            y1,y2 = ty[0],ty[1]
        m = (y2-y1)/(x2-x1)
        b = y2-(m*x2)
        xx,yy = np.cos(angp),np.sin(angp)
        mm = yy/xx
        xt = b/(mm-m)
        yt = mm*xt 
        fr = np.sqrt(xt*xt+yt*yt)/np.sqrt(x2*x2+y2*y2)
        x,y = fr*np.cos(tmang),fr*np.sin(tmang)

        self.x = np.vstack((self.x,x))
        self.y = np.vstack((self.y,y))

        F  = plt.figure()
        ax = F.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(int(self.n)+1):
            ax.plot(self.x[i],self.y[i],'k-',lw=2)
        ax.plot([x1,x2],[y1,y2],'go')
        ax.plot([0,xt],[0,yt],'r--')
        ax.plot(xt,yt,'ro')
        plt.show()

class FractalFlower(object):

    def __init__(self,N=6,xs=0.,ys=0.,npoints=1000,rad=1.):

        self.xs = xs
        self.ys = ys
        self.N = N
        self.npoints = npoints
        self.rad=rad
        
        cent = SimpleCircle(xc=xs,yc=ys,npoints=self.npoints,radius=self.rad)
        self.angs = np.linspace(0,np.pi*2.,N+1)[:-1]
        self.xc,self.yc = np.cos(self.angs),np.sin(self.angs)
        self.layer = 0
        self.values = {f'layer{self.layer}':{'coords':np.vstack((rad*cent.x,rad*cent.y)),
                                             'cents':np.array([self.xc,self.yc])}
                      }
        self.hi = np.max(rad*cent.x)

    def add_layer(self):
        
        self.layer+=1
        ext = 1./(2.*np.cos(self.angs[1]))
        sc = self.rad*self.layer*ext
        tx,ty = [],[]
        for i,ang in enumerate(self.angs):
            tx.append((self.rad*self.layer*self.xc[i])+self.xs)
            ty.append((self.rad*self.layer*self.yc[i])+self.ys)

            if self.layer > 1:
                try:
                    x1,x2 = (sc*self.xc[i])+self.xs,(sc*self.xc[i+1])+self.xs
                    y1,y2 = (sc*self.yc[i])+self.ys,(sc*self.yc[i+1])+self.ys
                    
                except:
                    x1,x2 = (sc*self.xc[i])+self.xs,(sc*self.xc[0])+self.xs
                    y1,y2 = (sc*self.yc[i])+self.ys,(sc*self.yc[0])+self.ys
                    
                if np.abs(x2-x1) >= 1.e-6:
                    tline = SimpleLine(x1,x2,y1,y2)
                    xs = np.linspace(x1,x2,self.layer+1)[1:-1]
                    ys = tline.get_y(xs)
                else:
                    xs = []
                    for i in range(self.layer-1): xs.append(x1)
                    ys = np.linspace(y2,y1,self.layer+1)[1:-1]
                for j in range(len(xs)):
                    tx.append(xs[j])
                    ty.append(ys[j])
        tmc = np.zeros((len(tx),2,self.npoints))
        for i in range(len(tx)):
            tcirc = SimpleCircle(xc=tx[i],yc=ty[i],npoints=self.npoints,radius=sc)
            tmc[i] = np.vstack((tcirc.x,tcirc.y))
            hi = np.max(tcirc.x)
            if hi > self.hi: self.hi = hi

        self.values[f'layer{self.layer}'] = {'coords':tmc}

        
        tx.append(tx[0])
        ty.append(ty[0])
        self.values[f'layer{self.layer}']['cents'] = np.vstack((tx,ty))

    def tmp(self,ax,c='k',minlw=1,maxlw=6,ilo=0.1,ihi=25.,palette=[0]):

        lfct = ilo/ihi
        hfct = ihi/ilo
        
        ks = list(self.values.keys())
        lws = np.linspace(minlw,maxlw,len(ks))
        for i,k in enumerate(ks):
            tmlw = lws[i]
            tmd = self.values[k]['coords']
            if len(palette) == 1:
                tmc = np.random.choice(['crimson','orange','gold','darkgreen','dodgerblue','violet'])
            else:
                tmc = palette[np.random.randint(0,palette.shape[0])]
            if k == 'layer0':
                ax.plot(tmd[0],tmd[1],'-',c=tmc,lw=tmlw,alpha=.1)
                ax.plot(tmd[0]*lfct,tmd[1]*lfct,'-',c=tmc,lw=tmlw,alpha=.2)
                ax.plot(tmd[0]*hfct,tmd[1]*hfct,'-',c=tmc,lw=tmlw,alpha=.2)
            else:
                for i in range(tmd.shape[0]):
                    ax.plot(tmd[i][0],tmd[i][1],'-',c=tmc,lw=tmlw,alpha=.1)
                    ax.plot(tmd[i][0]*lfct,tmd[i][1]*lfct,'-',c=tmc,lw=tmlw,alpha=.2)
                    ax.plot(tmd[i][0]*hfct,tmd[i][1]*hfct,'-',c=tmc,lw=tmlw,alpha=.2)

class Pyrograph():

    def __init__(self,hole=0.1,inner_frac=0.636,outer_R = 1.,xo=0.,yo=0.,nps=1000):
        self.npoints = nps
        self.R = outer_R
        self.r = self.R*inner_frac
        self.rho = self.r*hole

        dm = divmod(self.R,(1.-self.r))
        self.one_orb = 2.*np.pi*(dm[0]-1)
        if self.one_orb == 0.:
            self.one_orb = 2.*np.pi
        self.theta = 0.
        self.cx,self.cy = 0.,0.

        self.n = 0

    def one_orbit(self):

        theta = np.linspace(float(self.n)*self.one_orb,float(self.n+1)*self.one_orb,self.npoints)
        cx = (self.R-self.r)*np.cos(theta)
        cy = (self.R-self.r)*np.sin(theta)
        itheta = ((-1.)*(self.R-self.r)/self.r)*theta
        ix = self.rho*np.cos(itheta)
        iy = self.rho*np.sin(itheta)

        if self.n == 0.:
            self.x = cx+ix
            self.y = cy+iy
            self.n += 1
        else:
            self.x = np.vstack((self.x,cx+ix))
            self.y = np.vstack((self.y,cy+iy))
            self.n += 1

def testing():
    from color_tools import colorFader as cF
    
    F = plt.figure(frameon=False,dpi=150)
    F.set_size_inches(3,3)
    ax = plt.Axes(F,[0.,0.,1.,1.])
    ax.set_axis_off()
    ax.set_aspect('equal')
    F.add_axes(ax)

    norb = 175
    c1 = 'm'
    c2 = 'darkcyan'
    test = Pyrograph()
    t2 = Pyrograph(hole=0.63,inner_frac=0.635,outer_R = .38)
    for i in range(norb):
        test.one_orbit()
        t2.one_orbit()
    for i in range(norb):
        col=cF(c1,c2,float(i)/float(norb-1))
        c22 = cF('limegreen','tab:orange',float(i)/float(norb-1))
        ax.plot(test.x[i],test.y[i],'-',c=col,lw=2,alpha=1)
        ax.plot(t2.x[i],t2.y[i],'-',c=c22,alpha=.6,lw=1)
    plt.show()
                    
if __name__ == '__main__':

    """
    sz=SpinZoomer()
    for i in range(2): sz.add_one()
    sz.tmp()
    """
