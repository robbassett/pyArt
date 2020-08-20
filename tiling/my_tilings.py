import numpy as np
import matplotlib.pyplot as plt

from my_patches import *

# "Regular" Tilings center generators
class EqTriTi(object):

    def __init__(self,Nr,Nc,L=1.):
        hyp = (L/2.)/np.cos(np.pi/6.)
        yp = np.sqrt((hyp*hyp)-((L/2.)*(L/2.)))

        x = np.linspace(0.,Nc*(L/2.),Nc)
        y = np.linspace(0.,Nr*2*yp,Nr)
        xx,yy = np.meshgrid(x,y)
        angs = np.zeros(xx.shape)
        cx,cy = np.zeros(xx.shape),np.zeros(yy.shape)
        for i in range(xx.shape[0]):
            for j in range(yy.shape[1]):
                tx,ty = xx[i,j],yy[i,j]
                print(tx,ty)
                
                if (i+j)/2. == round((i+j)/2.):
                    #yy[i,j] += yp/2.
                    angs[i,j] = np.pi

        self.x = xx.ravel()
        self.y = yy.ravel()
        self.angs = angs.ravel()

eqcs = EqTriTi(8,8)

F = plt.figure()
ax = F.add_subplot(111)
ax.set_aspect('equal')
ax.plot(eqcs.x,eqcs.y,'ro')

for i in range(len(eqcs.x)):
    tc = [eqcs.x[i],eqcs.y[i]]
    if eqcs.angs[i] == 0:
        tri = EqTri(tc,rang=eqcs.angs[i])
        ptc = patches.PathPatch(tri.path,fc='r',ec='w',lw=1)
        ax.add_patch(ptc)

plt.show()
