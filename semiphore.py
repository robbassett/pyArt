import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
from matplotlib.path import Path
import matplotlib.patches as patches
from utils.color_tools import colorFader
import matplotlib.cm as cm

def mkptchs(cent,th1,L=1.,cshft=[0,0],r=.1):
    h = L/2.
    x = h*np.cos(th1)

    if r >= 0.5:
        xs = np.array([x,h,h,-x,cshft[0],x])
        ys = np.array([h,h,-h,-h,cshft[1],h])
    else:
        xs = np.array([h,h,-h,-h,cshft[0],h])
        ys = np.array([x,h,h,-x,cshft[1],x])
        
    vs = np.vstack((xs,ys))
    vs[0]+=cent[0]
    vs[1]+=cent[1]
    
    verts1 = [[vs[0][i],vs[1][i]] for i in range(len(xs))]
    codes1 = [Path.MOVETO]
    for i in range(len(xs)-2): codes1.append(Path.LINETO)
    codes1.append(Path.CLOSEPOLY)
    
    if r >= 0.5:
        xx = np.array([x,-h,-h,-x,cshft[0],x])
    else:
        xx = np.array([h,h,-h,-h,cshft[0],h])
        ys = np.array([x,-h,-h,-x,cshft[1],x])
        
    vv = np.vstack((xx,ys))
    vv[0]+=cent[0]
    vv[1]+=cent[1]
    
    verts2 = [[vv[0][i],vv[1][i]] for i in range(len(xx))]
    codes2 = [Path.MOVETO]
    for i in range(len(xx)-2): codes2.append(Path.LINETO)
    codes2.append(Path.CLOSEPOLY)

    return Path(verts1,codes1),Path(verts2,codes2),xs.max()+cent[0],ys.max()+cent[1]

row,col = 11,9
F = plt.figure(frameon=False,figsize=(col,row),dpi=200)
ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
ax.set_aspect('equal')
ax.set_axis_off()
F.add_axes(ax)
colors = ['w','k','tab:orange','tab:red','gold','dodgerblue']
mx,my = 0,0
for i in range(col):
    for j in range(row):
        tr = np.random.uniform()
        p1,p2,xm,ym = mkptchs([i*1.2,j*1.2],np.random.uniform(0.,np.pi),cshft=[.2,.2],r=tr)
        c1,c2 = np.random.choice(colors,2,replace=False)
        p1,p2 = patches.PathPatch(p1,fc=c1,ec='k',lw=2),patches.PathPatch(p2,fc=c2,ec='k',lw=2)

        ax.add_patch(p1)
        ax.add_patch(p2)

        mx = max([mx,xm])
        my = max([my,ym])
        
ax.set_xlim(-.7,mx+.2)
ax.set_ylim(-.7,my+.2)
plt.savefig('gallery/semiphore1.jpg')


F = plt.figure(frameon=False,figsize=(col,row),dpi=200)
ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
ax.set_aspect('equal')
ax.set_axis_off()
F.add_axes(ax)
colors = ['beige','silver','tab:pink','plum','palevioletred','lightcoral']
mx,my = 0,0
for i in range(col):
    for j in range(row):
        tr = np.random.uniform()
        tx,ty = 0.9*(i-(int(col/2)))/col,0.9*(j-int(row/2))/row
        p1,p2,xm,ym = mkptchs([i*1.2,j*1.2],np.random.uniform(0.,np.pi),cshft=[tx,ty],r=tr)
        c1,c2 = np.random.choice(colors,2,replace=False)
        p1,p2 = patches.PathPatch(p1,fc=c1,ec='k',lw=2),patches.PathPatch(p2,fc=c2,ec='k',lw=2)

        ax.add_patch(p1)
        ax.add_patch(p2)

        mx = max([mx,xm])
        my = max([my,ym])
        
ax.set_xlim(-.7,mx+.2)
ax.set_ylim(-.7,my+.2)
plt.savefig('gallery/semiphore2.jpg')


F = plt.figure(frameon=False,figsize=(col,row),dpi=200)
ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
ax.set_aspect('equal')
ax.set_axis_off()
F.add_axes(ax)
colors1 = [
    ['forestgreen','darkgreen','olivedrab','mediumseagreen'],
    ['darkorange','tab:orange','sandybrown','orange'],
    ['darkviolet','mediumorchid','blueviolet','violet']
]
colors2 = [
    ['salmon','lightcoral','lightpink','tab:red'],
    ['powderblue','paleturquoise','lightsteelblue','deepskyblue'],
    ['gold','y','khaki','lightyellow']
]
mx,my = 0,0
for i in range(col):
    tc1 = colors1[divmod(i,3)[0]]
    tc2 = colors2[divmod(i,3)[0]]
    for j in range(row):
        colors = [colorFader(tc1[k],tc2[k],j/(row-1)) for k in range(len(colors1))]
        tr = np.random.uniform()
        tx,ty = 0,0.45*j/(row-1)-.1
        p1,p2,xm,ym = mkptchs([i*1.2,j*1.2],np.random.uniform(0.,np.pi),cshft=[tx,ty],r=tr)
        c1,c2 = np.random.choice(colors,2,replace=False)
        p1,p2 = patches.PathPatch(p1,fc=c1,ec='k',lw=2),patches.PathPatch(p2,fc=c2,ec='k',lw=2)

        ax.add_patch(p1)
        ax.add_patch(p2)

        mx = max([mx,xm])
        my = max([my,ym])
        
ax.set_xlim(-.7,mx+.2)
ax.set_ylim(-.7,my+.2)
plt.savefig('gallery/semiphore3.jpg')
