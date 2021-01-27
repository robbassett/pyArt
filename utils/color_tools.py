import numpy as np
import matplotlib as mpl
import imageio
from PIL import Image
import matplotlib.pyplot as plt

class KmeansPalette(object):

    def __init__(self,pin,K=16):
        pim = np.asarray(Image.open(pin))
        if pim.shape[2] == 4:
            bt = np.zeros((pim.shape[0],pim.shape[1],3))
            bt[:,:,:] = pim[:,:,:-1]
            pim = np.copy(bt)
        X = insh = pim.reshape((pim.shape[0]*pim.shape[1],3))
        
        vals = np.random.uniform(0,255,int(K))
        self.centroids = np.copy(vals)
        for i in range(2): self.centroids=np.vstack((self.centroids,np.random.uniform(0,255,int(K))))
        self.centroids = self.centroids.T
        
        self.X  = X
        self.C  = np.zeros(X.shape[0])
        self.K  = K
        self.r  = 0
        self.loss = 0
        self.xin,self.yin = pim.shape[1],pim.shape[0]

    # Assign each pixel to the nearest color
    def assign_points(self):
        for i in range(len(self.C)):
            self.r=np.sqrt(np.sum((self.centroids-self.X[i])*(self.centroids-self.X[i]),axis=1))
            self.C[i] = np.where(self.r == np.min(self.r))[0][0]

    # Shift the colors towards the mean of matched pixels
    def move_centroids(self,lrate=1.):
        for i in range(self.K):
            t = np.where(self.C == i)[0]
            if len(t) > 0:
                true_cent = np.array([np.mean(self.X[:,0][t]),np.mean(self.X[:,1][t]),np.mean(self.X[:,2][t])])
                diff = self.centroids[i].astype(float)-true_cent
                self.centroids[i] -= (diff*lrate)
            else:
                # poor poor unused colors :'(
                self.centroids[i] = np.random.uniform(0,255)

    # Reconstruct the current image
    def remake_im(self,fnm):
        imout = np.zeros(self.X.shape).astype(np.uint8)
        for i in range(self.X.shape[0]):
            imout[i] = self.centroids[self.C[i].astype(np.uint8)]

        imout = imout.reshape((self.xin,self.yin,3))
        self.im = imout
        imout = Image.fromarray(imout)
        imout.save(f'./{fnm}.png')

    def output_palette(self,iters=20):
        for i in range(iters):
            self.assign_points()
            self.move_centroids()
        return self.centroids.astype(int)

    def display_palette(self):
        x = np.linspace(0,self.K,self.K)
        y = x*0.
        cs = []
        for i in range(len(x)):
           cs.append(mpl.colors.to_hex(self.centroids[i]/255.))
        
        F = plt.figure()
        ax = F.add_subplot(111)
        ax.scatter(x,y,s=self.K*100,c=cs)
        plt.show()

        
def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def sinuswitchcolor(length,periods,c1,c2,phase=0.):
    clt = np.linspace(0,2.*periods*np.pi,length)
    clt = (((-1.)*np.cos(clt+phase))/2.)+0.5
    
    return [colorFader(c1,c2,_c) for _c in clt]

def sinuflashcolor(length,periods,c1,c2,frate=10,phase=0):
    clt = np.linspace(0,2.*periods*np.pi,length)
    clt = (((-1.)*np.cos(clt+phase))/2.)+0.5
    clt = np.exp(frate*clt)
    clt/= np.max(clt)
        
    return [colorFader(c1,c2,_c) for _c in clt]
