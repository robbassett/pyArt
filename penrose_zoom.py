from matplotlib.pyplot import plot
from tiling.penrose import *
import imageio
from utils.color_tools import *
from utils.anim_tools import fade_frames
import numpy as np

FSIZE = [4,4.2]

def plot_all_tiles(T,xlim,ylim,dcols,kcols):
    F = plt.figure(frameon=False,figsize=FSIZE,dpi=250)
    ax = plt.Axes(F,[0.01,0.01,.98,.98])
    ax.set_aspect('equal')
    ax.set_axis_off()
    F.add_axes(ax)
    for t in T:
        cs = dcols if t.type == 'Dart' else kcols
        p = patches.PathPatch(t.path,fc=np.random.choice(cs),ec='k')
        ax.add_patch(p)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.fill_between([-2,2],2,-2,color='k')
    plt.savefig('frame.png')
    return imageio.imread('frame.png')

def inflate_all(T):
    tmt = []
    for t in T: tmt+=list(t.inflate())
    return tmt

a1 = 2*np.pi/5.
tiles = [Dart([0,0],np.pi/2.+a1*_,1) for _ in range(5)]
dcols1 = ['tab:orange','darkorange','orange']
dcols2 = ['tab:red','orangered','red']
kcols1 = ['tab:blue','royalblue','dodgerblue']
kcols2 = ['darkorchid','darkviolet','blueviolet']

npause = 8
nfade = 5
ninflate = 6
nframes = ninflate*npause*nfade + nfade*ninflate*4 + npause*2 + 1

def fade_frames(im1,im2,images,n_fade):
    alpha = np.linspace(0,1,n_fade)
    for _a in alpha[:-1]:
        tm_im = (1.-_a)*im1 + (_a)*im2
        imageio.imwrite('frame.png',tm_im)
        images.append(imageio.imread('frame.png'))

dcols = np.array([sinuswitchcolor(nframes,14,dcols1[_],dcols2[_]) for _ in range(3)]).T
kcols = np.array([sinuswitchcolor(nframes,10,kcols1[_],kcols2[_]) for _ in range(3)]).T
ind = 0
images = []
for i in range(ninflate):
    for j in range(npause):
        im1 = plot_all_tiles(tiles,[-1,1],[-1,1.1],dcols[ind],kcols[ind])
        im2 = plot_all_tiles(tiles,[-1,1],[-1,1.1],dcols[ind+1],kcols[ind+1])
        fade_frames(im1,im2,images,nfade)
        ind += 1
    tiles = inflate_all(tiles)
    im3 = plot_all_tiles(tiles,[-1,1],[-1,1.1],dcols[ind+1],kcols[ind+1])
    alpha = np.linspace(0,1,nfade*4)
    c = True
    for j in range(4*nfade):
        if j%nfade == 0:
            im3 = plot_all_tiles(tiles,[-1,1],[-1,1.1],dcols[ind+1],kcols[ind+1])
            if c:
                im4 = im1
                c = False
            else:
                im4 = im2
                c = True
        tm_im = (1.-alpha[j])*im4 + (alpha[j])*im3
        imageio.imwrite('frame.png',tm_im)
        images.append(imageio.imread('frame.png'))

for j in range(npause*2):
    im1 = plot_all_tiles(tiles,[-1,1],[-1,1.1],dcols[ind],kcols[ind])
    im2 = plot_all_tiles(tiles,[-1,1],[-1,1.1],dcols[ind+1],kcols[ind+1])
    fade_frames(im1,im2,images,nfade)
    ind += 1

imageio.mimsave('./penfade.mp4',images)