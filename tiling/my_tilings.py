import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from my_patches import *

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

# "Regular" Tilings center generators
class A2L(object):

    def __init__(self,Ngen,L=1.):
        l = (L/2.)/np.cos(np.pi/6.)
        xys = [[0.,0.]]
        dn = [0]
        angs = np.linspace(0.,2.*np.pi,7)[:-1]
        print(Ngen)
        for N in range(Ngen):
            print(N)
            cc = np.copy(xys)
            for i,coord in enumerate(cc):
                #print(coord)
                if dn[i] == 0:
                    dn[i] = 1
                    for ang in angs:
                        dx,dy = l*np.cos(ang),l*np.sin(ang)
                        if abs(dx) < 1.e-9: dx = 0.
                        if abs(dy) < 1.e-9: dy = 0.
                        tc = [coord[0]+dx,coord[1]+dy]
                        if tc not in xys:
                            xys.append(tc)
                            dn.append(0)

        coords = np.array(xys).T
        rang = np.pi/6.
        rmat = np.array([
            [np.cos(rang),(-1.)*np.sin(rang)],
            [np.sin(rang),np.cos(rang)]
        ])
        coords = np.dot(coords.T,rmat).T
        
        self.x = coords[0]
        self.y = coords[1]
        
class EqTriTi(object):

    def __init__(self,Nr,Nc,L=1.):
        ysp = L/2.*np.tan(np.pi/6.)
        cnt = 0
        x = np.zeros((Nr,Nc))
        y = np.zeros((Nr,Nc))
        angs = np.zeros((Nr,Nc))
        row = 0
        r = 0
        while row < Nr:
            if cnt not in [2,5]:
                ty = ysp*float(r)
                cx = np.arange(0.,Nc*L,L)
                if cnt in [1,3]:
                    cx+=L/2.
                x[row] = cx
                y[row] = cx*0.+ty
                if cnt in [1,4]:
                    angs[row] = np.pi
                row += 1
            r += 1
            cnt+=1
            if cnt == 6: cnt = 0

        self.x = x.ravel()
        self.y = y.ravel()
        self.angs = angs.ravel()       
        
L = 12.
Nr,Nc = 44,25
eqcs = EqTriTi(Nr,Nc,L=L)
c1,c2 = 'r','orange'

F = plt.figure()
ax = F.add_subplot(111)
ax.set_aspect('equal')

for i in range(len(eqcs.x)):
    tc = [eqcs.x[i],eqcs.y[i]]
    c1 = colorFader('r','orange',tc[1]/eqcs.y.max())
    c2 = colorFader('pink','darkcyan',tc[1]/eqcs.y.max())
    if eqcs.angs[i] == 0:
        tcol = colorFader(c1,c2,tc[0]/eqcs.x.max())
        tri = EqTri(tc,rang=eqcs.angs[i],L=L)
        ptc = patches.PathPatch(tri.path,fc=tcol,ec='None')
        ax.add_patch(ptc)
    else:
        tcol = colorFader(c2,c1,tc[0]/eqcs.x.max())
        tri = EqTri(tc,rang=eqcs.angs[i],L=L)
        ptc = patches.PathPatch(tri.path,fc=tcol,ec='None')
        ax.add_patch(ptc)

ax.set_xlim(eqcs.x.min(),eqcs.x.max())
ax.set_ylim(eqcs.y.min(),eqcs.y.max())
plt.show()

