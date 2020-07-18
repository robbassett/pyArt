import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def sinuswitchcolor(length,periods,c1,c2):
    clt = np.linspace(0,2.*periods*np.pi,length)
    clt = (((-1.)*np.cos(clt))/2.)+0.5
    
    return [colorFader(c1,c2,_c) for _c in clt]

class SinuCirc(object):

    def __init__(self,radius,inner_omega):
        self.r = radius
        self.om= inner_omega
        
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
          

    def Display(self,ax,xoff=0.,yoff=0.,alph=0.5,c1='k',c2='r',c3='w'):

        ax.fill_between(self.x1+xoff,self.y1+yoff,((-1.)*self.y1)+yoff,color=c3)
        ax.plot(self.x+xoff,self.y+yoff,'-',c=c1,lw=1,alpha=alph)
        ax.plot(self.x+xoff,((-1.)*self.y)+yoff,'-',c=c1,lw=1,alpha=alph)
        ax.plot(self.x1+xoff,self.y1+yoff,'-',c=c2,lw=4)
        ax.plot(self.x1+xoff,((-1.)*self.y1)+yoff,'-',c=c2,lw=4)
        ax.plot(self.x+xoff,((-1.)*self.yi)+yoff,'-',c=c2,lw=2.)

    def Check(self,th,nsp):

        F  = plt.figure()
        ax = F.add_subplot(111)
        for i in range(nsp):
            self.Spin(i*th)
            ax.plot(self.x,self.y,'k-',lw=2,alpha=1./float(nsp))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.show()
