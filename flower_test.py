import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import glob
import os
from PIL import Image

from art_objs import *
from color_tools import *
from iseeall import draw_eye

class NewFlower(object):

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
                ax.plot(tmd[0]*lfct,tmd[1]*lfct,'-',c=tmc,lw=tmlw,alpha=.1)
                ax.plot(tmd[0]*hfct,tmd[1]*hfct,'-',c=tmc,lw=tmlw,alpha=.1)
            else:
                for i in range(tmd.shape[0]):
                    ax.plot(tmd[i][0],tmd[i][1],'-',c=tmc,lw=tmlw,alpha=.1)
                    ax.plot(tmd[i][0]*lfct,tmd[i][1]*lfct,'-',c=tmc,lw=tmlw,alpha=.1)
                    ax.plot(tmd[i][0]*hfct,tmd[i][1]*hfct,'-',c=tmc,lw=tmlw,alpha=.1)

class FlowerTest(PowerFlower):

    def __init__(self,N=6,xs=0.,ys=0.,npoints=1000,rad=1.):
        super().__init__(N=N,xs=xs,ys=ys,npoints=npoints,rad=rad)

    def tmp(self,ax,c='k'):

        for k in self.values.keys():
            tmd = self.values[k]['coords']

            if k == 'layer0':
                ax.plot(tmd[0],tmd[1],'-',c=c,lw=.5)
            else:
                for i in range(tmd.shape[0]):
                    ax.plot(tmd[i][0],tmd[i][1],'-',c=c,lw=.5)

    def pmt(self,ax,c='k'):
    
        
        for k in self.values.keys():
            tmd = self.values[k]['cents']
            ax.plot(tmd[0],tmd[1],'o',c=c,ms=4)

    def tmt(self,ax):

        for k in self.values.keys():
            if k == 'layer0': pass
            else:
                for i in range(self.values[k]['coords'].shape[0]):
                    
                    color = np.random.choice(['crimson','orange','gold','darkgreen','dodgerblue','violet'])
                    cy = self.values[k]['cents'][1][i]
                    crc = self.values[k]['coords'][i]
                    t1 = np.where(crc[1] >= cy)[0]
                    t2 = np.where(crc[1] < cy)[0]
                    try:
                        ax.fill_between(crc[0][t1],crc[1][t1],crc[1][t2],alpha=.3,color=color)
                    except:
                        ax.fill_between(crc[0][t1[1:-1]],crc[1][t1[1:-1]],crc[1][t2],alpha=.3,color=color)

"""
F  = plt.figure()
ax = F.add_subplot(111)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

angs = np.linspace(0,np.pi*2.,8)[:-1]
xs = 9.*np.cos(angs)
ys = 9.*np.sin(angs)
rads = np.linspace(0.5,1.5,8)
for j in range(len(xs)):
    d = FlowerTest(N=j+3,xs=xs[j],ys=ys[j],rad=rads[j])
    for i in range(2):
        d.add_layer()

    d.tmt(ax)
    d.tmp(ax,c='w')
    #d.pmt(ax,c='gray')
plt.show()


images = []
for i in range(50):
    F = plt.figure()
    ax = F.add_subplot(111)
    d = FlowerTest(N=6)
    for i in range(3):
        d.add_layer()
    d.tmt(ax)
    d.tmp(ax,c='w')
    plt.savefig(f'tmp.png')
    plt.close(F)
    images.append(imageio.imread('tmp.png'))
imageio.mimsave('test.gif', images)
"""
pim = np.asarray(Image.open('tmp/pastel.png'))
if pim.shape[2] == 4:
    bt = np.zeros((pim.shape[0],pim.shape[1],3))
    bt[:,:,:] = pim[:,:,:-1]
pim = np.copy(bt)

insh = pim.reshape((pim.shape[0]*pim.shape[1],3))
tst  = KmeansPalette(insh,K=16,xin=pim.shape[0],yin=pim.shape[1])
for i in range(20):
    tst.assign_points()
    tst.move_centroids()
palette = tst.centroids/255.
                        
d=NewFlower(N=17)
for i in range(23):
    d.add_layer()

    
F=plt.figure(figsize=(5,5),dpi=100)
ax=F.add_subplot(111)
lo,hi = .05,d.hi
d.tmp(ax,minlw=2,maxlw=4,ilo=lo,ihi=hi,palette=palette)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
nframe = 75
lms = np.linspace(np.log10(lo),np.log10(hi),nframe)
lms = 10.**(lms)
sml = np.flip(lms)
images=[]
for i in range(nframe-1):
    ax.set_xlim((-1.)*sml[i],sml[i])
    ax.set_ylim((-1.)*sml[i],sml[i])
    plt.tight_layout()
    plt.savefig('./frame.png')
    images.append(imageio.imread('./frame.png'))
imageio.mimsave('./test.gif',images)
