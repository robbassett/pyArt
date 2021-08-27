import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

def colorFader(c1,c2,mix=0): 
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1))+mix*np.array(mpl.colors.to_rgb(c2)))

def one_square(cx=0.5):
    s = [[-.5+cx,.5,-.5+cx,-.5],[-.5,-.5+cx,.5,-.5+cx]]
    sides = np.random.choice([0,1,2,3],replace=False,size=2)
    return np.array([s[0][_] for _ in sides]),np.array([s[1][_] for _ in sides])

NSQ,FSIZE,NFRAME = 15,(6,8),50
SEED = 212572018

images=[]
ccxx = list(np.linspace(0,1,int(NFRAME/2)))
for CX in ccxx[:-1]+ccxx[::-1][:-1]:
    F = plt.figure(frameon=False,figsize=FSIZE,dpi=250)
    ax = plt.Axes(F,[0.01,0.01,.98,.98])
    ax.set_aspect('equal')
    ax.set_axis_off()
    F.add_axes(ax)
    np.random.seed(SEED)
    for i in range(NSQ):
        for j in range(int(NSQ*1.333)):
            x,y = one_square(cx=CX)
            for c2,m in zip(['silver','lightsteelblue'],['-','o']): 
                ax.plot(x+i,y+j,m,lw=4,ms=4,c=colorFader('k',c2,mix=(0.5+max(y)+j)/int(NSQ*1.333)))

    plt.savefig('frame.png')
    plt.close()
    images.append(imageio.imread('frame.png'))
imageio.mimsave('./molnar1.gif',images)