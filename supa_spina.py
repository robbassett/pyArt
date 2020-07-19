import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import glob
import os

from art_objs import *
from color_tools import *
from iseeall import draw_eye

class SpinBoi(SinuCirc):

    def __init__(self,radius,inner_omega):
        super().__init__(radius,inner_omega)

    def Display(self,ax,xoff=0.,yoff=0.,alph=0.5,c1='k',c2='r',c3='w'):

        ax.fill_between(self.x1+xoff,self.y1+yoff,((-1.)*self.y1)+yoff,color=c3)
        ax.plot(self.x+xoff,self.y+yoff,'-',c=c1,lw=1,alpha=alph)
        ax.plot(self.x+xoff,((-1.)*self.y)+yoff,'-',c=c1,lw=1,alpha=alph)
        ax.plot(self.x1+xoff,self.y1+yoff,'-',c=c2,lw=4)
        ax.plot(self.x1+xoff,((-1.)*self.y1)+yoff,'-',c=c2,lw=4)
        ax.plot(self.x+xoff,((-1.)*self.yi)+yoff,'-',c=c2,lw=2.)

if __name__ == '__main__':

    chk = glob.glob('frames')
    if len(chk) == 0:
        os.mkdir('frames')
    
    frames = 150
    olo,ohi = 30.,150.
    osp = (ohi-olo)/2.
    
    ths = np.linspace(np.pi/100.,np.pi,frames)
    oms = np.concatenate((np.linspace(1./5.,1./np.sqrt(300.),frames/2),np.linspace(1./np.sqrt(300.),1./5.,frames/2)))
    
    omt = np.linspace(0,2.*np.pi,frames)
    oms = ((-1.)*np.cos(omt)*osp)+osp+olo
    tht = np.linspace(0,2.*np.pi,frames)
    ths = ((-1.)*np.cos(tht)*np.pi/4.)+(np.pi/4.)

    cl1 = sinuswitchcolor(frames,3,'darkorange','dodgerblue')
    cl2 = sinuswitchcolor(frames,6,'crimson','forestgreen')
    cl3 = sinuswitchcolor(frames,9,'gold','lightyellow')

    aeye = np.linspace(0,20,frames/2)
    teye = np.linspace(0,5,frames/2)
    jeye = np.linspace(0,8.,frames/2)+5.
    teye = np.exp((-1.)*teye)
    aeye = np.exp((-1.)*aeye)
    reye = np.copy(aeye)*5.
    aeye = np.concatenate((np.flip(aeye),teye))
    reye = np.concatenate((np.flip(reye),jeye))
    yeye = np.zeros(int(frames/2))-3.
    yeye = np.concatenate((yeye,np.linspace(-3.,0.,frames/2)))
    
    fnm = 1
    images = []
    for j,om in enumerate(oms):
        F  = plt.figure(figsize=(4,4),dpi=100)
        ax = F.add_subplot(111)
        for i in range(4):
            x=SpinBoi(5.,om)
            x.Spin(ths[j]*i)
            x.Display(ax,alph=0.25,c1=cl1[j],c2=cl2[j],c3=cl3[j])

        draw_eye(ax,[0.,yeye[j]],reye[j],aeye[j],c1=cl2[j],c2=cl1[j])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-5.2,5.2)
        ax.set_ylim(-5.2,5.2)
        ax.set_aspect('equal')
        
        plt.savefig(f'frames/frame{fnm}.png')
        plt.close(F)
        images.append(imageio.imread(f'frames/frame{fnm}.png'))
        fnm+=1
    
    imageio.mimsave('supa_spina.gif', images)
    import shutil
    shutil.rmtree('./frames')
