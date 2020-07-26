import numpy as np
import matplotlib.pyplot as plt

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

    
    
