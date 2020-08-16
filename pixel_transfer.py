import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

# Life on the farm can be rough, will your pixels be
# enough to get you through the hard winter?
class PixelFarm():

    def __init__(self,imin,pixsize=3):
        tmim = Image.open(imin)
        self.im = np.array(tmim.getdata()).reshape((tmim.height,tmim.width,3))
        self.pxs = pixsize

    def get_N_pix(self,N):

        self.N = N
        X,Y = self.im.shape[0]-self.pxs,self.im.shape[1]-self.pxs
        self.pix = np.zeros((N*4,self.pxs,self.pxs,3)).astype(int)
        ind = 0
        for i in range(N):
            xc,yc = np.random.randint(0,X),np.random.randint(0,Y)
            tmp = self.im[xc:xc+self.pxs,yc:yc+self.pxs,:]
            self.pix[ind] = tmp
            ind+=1
            for i in range(3):
                tmp = np.rot90(tmp,axes=(0,1))
                self.pix[ind] = tmp
                ind+=1

# Always numpy, always.
def load_to_np(im):
    tmim = Image.open(im)
    return np.array(tmim.getdata()).reshape((tmim.height,tmim.width,3))

# Make your image from your image, its great fun!
def transpix(image,source_image,npix=50,spixsize=10):
    
    im = load_to_np(image)
    dmx = divmod(im.shape[0],spixsize)
    dmy = divmod(im.shape[1],spixsize)

    imo = im[dmx[1]:,dmy[1]:]
    sx,sy = [],[]
    for i in range(dmx[0]):
        for j in range(dmy[0]):
            sx.append(spixsize*i)
            sy.append(spixsize*j)

    inds = np.linspace(0,len(sx)-1,len(sx)-1).astype(int)
    np.random.shuffle(inds)
    cnt = 0
    src = PixelFarm(source_image,pixsize=spixsize)
    src.get_N_pix(npix)
    for ind in inds:
        tx,ty = sx[ind],sy[ind]
            
        cnt+=1
        if cnt == 20:
            src = PixelFarm(source_image,pixsize=spixsize)
            src.get_N_pix(npix)
            cnt = 0
            
            

        imclip = imo[tx:tx+spixsize,ty:ty+spixsize,:]
        losses = np.zeros(npix*4)

        for n in range(npix*4):
            losses[n] = np.sum(np.abs(imclip-src.pix[n]))
        t = np.where(losses == losses.min())[0]
        if len(t) > 1:
            t = np.random.choice(t)
        imo[tx:tx+spixsize,ty:ty+spixsize,:] = src.pix[t]

    return imo

def save_frame(arr,fname='frame.png',mxsz=4.):
    sh = arr.shape
    sh = np.array([sh[1],sh[0]])
    fc = mxsz/sh.max()
    
    F = plt.figure(frameon=False,dpi=150)
    F.set_size_inches(sh[0]*fc,sh[1]*fc)
    ax = plt.Axes(F,[0.,0.,1.,1.])
    ax.set_axis_off()
    F.add_axes(ax)
    ax.imshow(arr,interpolation='None',aspect='auto')
    plt.savefig(fname)
