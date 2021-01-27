import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

class EqTriTi2(object):

    def __init__(self,Nr,Nc,L=1.,fr=0.3):
        l = fr*L
        dy,dx = l*(np.sqrt(3.)/2.),L-(l/2.)
        Dx,Dy = L*(0.5-fr),L*(np.sqrt(3.)/2.)
        lL,ll = L*np.tan(np.pi/6.)/2.,l*np.tan(np.pi/6.)/2.
        P = (1.-fr)*L
        lP = P*np.tan(np.pi/6.)/2.

        x = np.zeros((Nr,Nc))
        y = np.zeros((Nr,Nc))
        N = 0
        for j in range(Nr):
            xs = float(j)*Dx
            ys = float(j)*Dy
            tmrm = divmod(j,5)[0]
            dxe,dye = tmrm*dx,tmrm*dy
            for i in range(Nc):
                tmdm = divmod(i,3)[0]
                edx,edy = tmdm*Dx,tmdm*Dy
                x[j,i] = xs+float(i)*dx-edx-dxe
                y[j,i] = ys+float(i)*dy-edy-dye

        self.x = x.ravel()
        self.y = y.ravel()
        self.x2 = self.x+((L-l)/2.)
        self.y2 = self.y+(ll+lL)
        self.x3 = self.x-((L-P)/2.)
        self.y3 = self.y+(lP+lL)

class EQt2g1():

    def __init__(self,x0,y0,L=1.,fr=0.34,mch=0.5,lch=0.5):
        l = fr*L
        dy,dx = l*(np.sqrt(3.)/2.),L-(l/2.)
        Dx,Dy = L*(0.5-fr),L*(np.sqrt(3.)/2.)
        lL = L*np.tan(np.pi/6.)/2.
        mL = (L/2.)/np.cos(np.pi/6.)
        ll = l*np.tan(np.pi/6.)/2.
        ml = (l/2.)/np.cos(np.pi/6.)
        P = (1.-fr)*L
        lP = P*np.tan(np.pi/6.)/2.
        mP = (P/2.)/np.cos(np.pi/6.)

        
        self.x = x0
        self.y = y0

        self.x2 = []
        self.y2 = []
        for i in range(3):
            p = np.random.uniform()
            if p > lch:
                if i == 0:
                    self.x2.append(x0+((L-l)/2.))
                    self.y2.append(y0+(ll+lL))
                if i == 1:
                    self.x2.append(x0-(L/2.))
                    self.y2.append(y0+lL-ml)
                if i == 2:
                    self.x2.append(x0+(l/2.))
                    self.y2.append(y0-mL+ll)

        self.x3 = []
        self.y3 = []
        for i in range(3):
            p = np.random.uniform()
            if p > mch:
                if i == 0:
                    self.x3.append(x0-((L-P)/2.))
                    self.y3.append(y0+(lP+lL))
                if i == 1:
                    self.x3.append(x0-(P/2.))
                    self.y3.append(y0-mL+lP)
                if i == 2:
                    self.x3.append(x0+(L/2.))
                    self.y3.append(y0+lL-mP)
        

class EQT2_generative():

    def __init__(self,N,L=1.,fr=0.34,bch=0.5):
        l = fr*L
        dy,dx = l*(np.sqrt(3.)/2.),L-(l/2.)
        Dx,Dy = L*(0.5-fr),L*(np.sqrt(3.)/2.)
        lL,ll = L*np.tan(np.pi/6.)/2.,l*np.tan(np.pi/6.)/2.

        self.x,self.y = [0.],[0.]
        self.x2,self.y2 = [],[]
        self.x3,self.y3 = [],[]
        done = []
        n = 0
        pd = [1,0,-1,-1,0,1]
        pD = [0,1,1,0,-1,-1]
        while n <= N:
            n+=1
            for E in range(len(self.x)):
                if E not in done:
                    done.append(E)
                    tx,ty = self.x[E],self.y[E]
                    tmE = EQt2g1(tx,ty,L=L,fr=fr,lch=.5,mch=.7)
                    for i in range(len(tmE.x2)):
                        tx2,ty2 = round(tmE.x2[i],3),round(tmE.y2[i],3)
                        if tx2 not in self.x2 and ty2 not in self.y2:
                            self.x2.append(tx2)
                            self.y2.append(ty2)
                    for i in range(len(tmE.x3)):
                        tx3,ty3 = round(tmE.x3[i],3),round(tmE.y3[i],3)
                        if tx3 not in self.x3 and ty3 not in self.y3:
                            self.x3.append(tx3)
                            self.y3.append(ty3)
                        
                    for i in range(6):
                        p = np.random.uniform()
                        if p > bch:
                            tdx = dx*pd[i]+Dx*pD[i]
                            tdy = dy*pd[i]+Dy*pD[i]
                            tX,tY = round(tx+tdx,3),round(ty+tdy,3)
                            if tX not in self.x and tY not in self.y:
                                self.x.append(tX)
                                self.y.append(tY)

        

if __name__ == '__main__':
    
    from my_patches import *

    FR = .25
    L = 200.
    Nr,Nc = 5,8
    eqcs = EqTriTi2(Nr,Nc,L=L,fr=FR)
    c1,c2 = 'r','orange'
    
    
    F = plt.figure(frameon=False,figsize=(5.,5.),dpi=180)
    ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
    ax.set_aspect('equal')
    ax.set_axis_off()
    F.add_axes(ax)

    for i in range(len(eqcs.x)):
        tc = [eqcs.x[i],eqcs.y[i]]
        tri = EqTri(tc,L=L,rang=np.pi)
        ptc = patches.PathPatch(tri.path,fc='r',ec='None')
        ax.add_patch(ptc)

        tc = [eqcs.x2[i],eqcs.y2[i]]
        tri = EqTri(tc,L=L*FR)
        ptc = patches.PathPatch(tri.path,fc='b',ec='None')
        ax.add_patch(ptc)
        
        tc = [eqcs.x3[i],eqcs.y3[i]]
        tri = EqTri(tc,L=L*(1.-FR))
        ptc = patches.PathPatch(tri.path,fc='gold',ec='None')
        ax.add_patch(ptc)

    ax.set_xlim(eqcs.x.min()*1.2,eqcs.x.max()*1.2)
    ax.set_ylim(eqcs.y.min()*1.2,eqcs.y.max()*1.2)
    plt.savefig('./tmp.png')
    
    """
    for I in range(15):
        FR = np.random.uniform()*0.5+0.2
        tst = EQT2_generative(2,fr=FR,L=L,bch=.5)
        F = plt.figure()
        ax = F.add_subplot(111)
        ax.set_aspect('equal')
        for i in range(len(tst.x)):
            tc = [tst.x[i],tst.y[i]]
            tri = EqTri(tc,L=L,rang=np.pi)
            ptc = patches.PathPatch(tri.path,fc='r',ec='None')
            ax.add_patch(ptc)
        
        for i in range(len(tst.x2)):
            tc = [tst.x2[i],tst.y2[i]]
            tri = EqTri(tc,L=L*FR)
            ptc = patches.PathPatch(tri.path,fc='b',ec='None')
            ax.add_patch(ptc)
    
        for i in range(len(tst.x3)):
            tc = [tst.x3[i],tst.y3[i]]
            tri = EqTri(tc,L=L*(1.-FR))
            ptc = patches.PathPatch(tri.path,fc='gold',ec='None')
            ax.add_patch(ptc)

        ax.set_xlim(np.min(tst.x)*1.3,np.max(tst.x)*1.3)
        ax.set_ylim(np.min(tst.y)*1.3,np.max(tst.y)*1.3)
        plt.show()
        


    import imageio
    
    tst = EQT2_generative(5,fr=FR,bch=.5)

    xs = np.linspace(-5,5,80)
    yx = np.linspace(0,8.*np.pi,80)
    ys = np.cos(yx)/2.
    images=[]
    for k in range(80):
        F = plt.figure(frameon=False,figsize=(5.,5.),dpi=180)
        ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
        ax.set_aspect('auto')
        ax.set_axis_off()
        F.add_axes(ax)
        ax.fill_between([-10,10],10,-10,color='w')
        ax.set_aspect('equal')
        for i in range(len(tst.x)):
            tc = [tst.x[i]+xs[k],tst.y[i]+ys[k]]
            tri = EqTri(tc,L=L,rang=np.pi)
            ptc = patches.PathPatch(tri.path,fc='r',ec='None')
            ax.add_patch(ptc)
        
        for i in range(len(tst.x2)):
            tc = [tst.x2[i]+xs[k],tst.y2[i]+ys[k]]
            tri = EqTri(tc,L=L*FR)
            ptc = patches.PathPatch(tri.path,fc='b',ec='None')
            ax.add_patch(ptc)
    
        for i in range(len(tst.x3)):
            tc = [tst.x3[i]+xs[k],tst.y3[i]+ys[k]]
            tri = EqTri(tc,L=L*(1.-FR))
            ptc = patches.PathPatch(tri.path,fc='gold',ec='None')
            ax.add_patch(ptc)

        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        plt.savefig('./tmp.png')

        im = imageio.imread('tmp.png')
        wood = imageio.imread('../random/wood.jpg')
        wood = wood[:,:900,:]
        back = np.zeros((wood.shape[0],wood.shape[1],4))+255.
        back[:,:,:-1] = wood
        red = np.array([255.,0.,0.,255.])
        rdiff = np.abs(im-red).sum(axis=2)
        r,c = np.where(rdiff < 50)

        im[r,c,:] = back[r,c,:]
        imageio.imwrite('frame.png',im)
        images.append(imageio.imread('frame.png'))
    imageio.mimsave('./test.mp4',images)
    
    """
