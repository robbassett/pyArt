import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba as rgba
from collections import deque
import imageio

from matplotlib.path import Path
import matplotlib.patches as patches

import utils.color_tools as ct

import fourierier as Fboi

def c(im,X,Y,cloud_deq,t,bw=10):
    c = [0.5,2-(.02*t)]
    r = np.sqrt((X-c[0])*(X-c[0]) + (Y-c[1])*(Y-c[1]))
    row,col = np.where(r < .3)
    for i in range(row.shape[0]):
        alph = max([0,(row[i]-X.shape[1])/X.shape[1]])
        tmc = alph*np.array([1.,1.,1.,1.]) + (1-alph)*im[row[i],col[i]]
        im[row[i],col[i]] = tmc
    return im

def clouds(im,X,Y,cloud_deq,t,bw=.2):

    # cloud = [t0,c_omeg,c_phase]
    fl = 0
    for c,cloud in enumerate(cloud_deq):
        y = 2-.02*(t-cloud[0])
        g = .08 + 0.01*(t-cloud[0])
        c = 0.5+.2*np.sin((t-cloud[0])*cloud[1] + cloud[2])
        x1,x2 = c-(g/2),c+(g/2)
        r1 = np.sqrt((X-x1)*(X-x1) + (Y-y)*(Y-y))
        r2 = np.sqrt((X-x2)*(X-x2) + (Y-y)*(Y-y))
        
        tbw = min([bw,bw*y/2.2])
        row,col = np.where((r1 <= tbw/2)|(r2 <= tbw/2)|((X <= x1)&(np.abs(Y-y) <= tbw/2))|((X >= x2)&(np.abs(Y-y) <= tbw/2)))
    
        for i in range(row.shape[0]):
            alph = max([0,(row[i]-0.7*X.shape[1])/X.shape[1]])
            tmc = alph*np.array([1.,1.,1.,1.]) + (1-alph)*im[row[i],col[i]]
            im[row[i],col[i]] = tmc
            
    if (t+5)%10 == 0:
        tmc = cloud_deq[0]
        tmc[0] = t + 5
        cloud_deq.append(tmc)
    return im

def circ_grad(im,X,Y,cent,R,c1,c2):
    
    c1 = np.array(rgba(c1))
    c2 = np.array(rgba(c2))

    r = np.sqrt((X-cent[0])*(X-cent[0]) + (Y-cent[1])*(Y-cent[1]))
    
    row,col = np.where(r <= R)
    rr = [row.min(),row.max()]
    cc = [col.min(),col.max()]

    for i in range(row.shape[0]):
        mix = (row[i]-rr[0])/(rr[1]-rr[0])
        tmc = mix*c1 + (1.-mix)*c2
        im[row[i],col[i]] = tmc

    return im

def mountain(X,cent,sigma,h,peak,n=10):
    x = np.linspace(0,X.shape[1]-1,n)
    y = np.exp((-1)*(x-cent)*(x-cent)/(2*sigma*sigma))
    y *= (peak-h)/y.max()
    y += peak + np.random.normal(scale=15,size=n)

    return x,y

def staff(ang,x0,ml,pl,pn):
    x = [x0,x0+ml*np.cos(ang)]
    y = [0,ml*np.sin(ang)]
    xf,yf = [],[]
    xs,ys = [],[]
    p0 = [x[1],y[1]]

    for _ in [0.6,0.7,0.8]:
        xs.append(x0+_*ml*np.cos(ang))
        ys.append(_*ml*np.sin(ang))

    pngs = np.linspace(np.pi/2.,2.5*np.pi,pn+1)
    for p in pngs:
        tx,ty = pl*np.cos(p+ang),pl*np.sin(p+ang)
        xf.append(p0[0]+pl*0.7*np.cos(p+ang))
        yf.append(p0[1]+pl*0.7*np.sin(p+ang))
        x+=[p0[0]+tx,p0[0]]
        y+=[p0[1]+ty,p0[1]]

    return x,y,xf,yf,xs,ys
    
    

x = np.linspace(0,1,1000)
y = np.linspace(0,2,2000)

X,Y = np.meshgrid(x,y)
R = 0.4
im = np.zeros((*X.shape,4))
#c1 = np.array(rgba('tab:red'))
c1 = np.array(rgba('skyblue'))
c2 = np.array(rgba('crimson'))
    
for i in range(X.shape[0]):
    mix = np.power(i,1.02)/(X.shape[0]-1)
    #mix = mix*mix*mix
    im[i,:] = mix*c1 + (1-mix)*c2

cent=(0.5,0.66)
im = circ_grad(im,X,Y,cent,0.41,'tab:red','crimson')
im = circ_grad(im,X,Y,cent,0.4,'red','crimson')
cent=(0.5,0.55)
im = circ_grad(im,X,Y,cent,0.2,'orangered','crimson')
cent=(0.5,0.44)
im = circ_grad(im,X,Y,cent,0.1,'tomato','crimson')
im/=im.max()

frames = 25
images = []

ml = 7
cldq = deque(maxlen=ml)
for i in range(ml):
    t0 = 10*i - 10*(ml-1)
    ph = np.random.uniform(0,np.pi)
    om = np.random.uniform(0.05,0.15)
    cldq.append([t0,om,ph])

xm,ym = mountain(X,300,190,75,280,n=23)
xs,ys,xf,yf,xsf,ysf = staff(7/17*np.pi,500,715,125,3)

fBOI = Fboi.FourierDrongo(Nterms=9,sym=3)

nc = 0
f = 0
hcyc = 38
fbomega = np.pi/hcyc
nframes = 12*hcyc
scl = 99.4
dph = np.pi/10
ol0 = ct.sinuflashcolor(hcyc,1,'k','crimson',frate=2,phase=0)
ol1 = ct.sinuflashcolor(hcyc,1,'k','tomato',frate=2,phase=3*dph)
ol2 = ct.sinuflashcolor(hcyc,1,'k','orangered',frate=2,phase=2*dph)
ol3 = ct.sinuflashcolor(hcyc,1,'k','tab:red',frate=2,phase=dph)
fcnt = 0
glitch = []
gord = [0,1,2,1,0,-1,-2,-3,-4,-8,-9,-2,-1]
gg = 0
while True:
    fcnt+=1
    
    if nframes - f > 13:
        cim = clouds(np.copy(im),X,Y,cldq,f)
        if f < 10 or nframes-f < 20:
            glitch.append(cim)
    else:
        cim = glitch[gord[gg]]
        gg+=1
    #cim = clouds(np.copy(im),X,Y,cldq,f)

    F = plt.figure(frameon=False,figsize=(2,4),dpi=200)
    ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
    ax.set_aspect('equal')
    ax.set_axis_off()
    F.add_axes(ax)

    ax.imshow(cim,origin='lower')
    ax.fill_between(xm,ym,color='k',zorder=9999)
    ax.plot(xs,ys,'k-',lw=3,zorder=10000)

    fBOI.make_points(phase=f*fbomega)
    fBOI.add_vertices()
    fBOI.make_patches()
    if f == 0:
        mxr = max(fBOI.prs)
        
    lee = np.array(fBOI.pls)
    tord = (-lee).argsort()
    
    for k in range(len(fBOI.pxs)):
        x,y = fBOI.pxs[tord[k]],fBOI.pys[tord[k]]
        rrr = scl*fBOI.prs[tord[k]]/mxr
        if rrr > 1.: rrr = 1.
        ang = (fBOI.plt[tord[k]])/(2.*np.pi)
        pvrts = []
        for i in range(len(x)): pvrts.append((scl*x[i]+xs[-1],scl*y[i]+ys[-1]))
        cds = [Path.MOVETO]
        for i in range(len(x)-2): cds.append(Path.LINETO)
        cds.append(Path.CLOSEPOLY)
        ptch = patches.PathPatch(Path(pvrts,cds),fc='k',ec='None',zorder=10002+k)
        ax.add_patch(ptch)

    I = f%len(ol0)
    ax.plot(scl*fBOI.x+xs[-1],scl*fBOI.y+ys[-1],'-',color=ol0[I],lw=0.75,zorder=20000)
    ax.plot(xf,yf,'o',ms=3,c=ol0[I],mec='k',zorder=10001)
    ax.plot(xsf[0],ysf[0],'o',ms=1.5,c=ol1[I],zorder=10001)
    ax.plot(xsf[1],ysf[1],'o',ms=1.5,c=ol2[I],zorder=10001)
    ax.plot(xsf[2],ysf[2],'o',ms=1.5,c=ol3[I],zorder=10001)

    
    plt.savefig('frame.png')
    plt.close(F)
    images.append(imageio.imread(f'frame.png'))
    
    #if (f-5)%10 == 0 and f != 0: nc+=1
    #if nc == 8:
    #    break
    if fcnt+1 == nframes:
        break
    f+=1
    
imageio.mimsave('gallery/A5ioPp0.0.mp4', images)
