import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import PatchCollection

plt.rcParams["font.family"] = "serif"

def g2d(x,y,x0,y0,sig=0.04):
    return np.exp((-1.)*(((x-x0)*(x-x0))+((y-y0)*(y-y0)))/(2.*sig*sig))

def border(mxr,n=50,fiddle=0.01):
    xs = np.linspace(-1.1*mxr,1.1*mxr,n)
    t = np.where(np.abs(xs) <= mxr)[0]
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y1[t] = np.sqrt(mxr*mxr - xs[t]*xs[t])+np.random.normal(scale=fiddle,size=len(t))
    y2[t] = np.sqrt(mxr*mxr - xs[t]*xs[t])+np.random.normal(scale=fiddle,size=len(t))
    xs[t] += np.random.normal(scale=fiddle,size=len(t))
    return xs,y1,y2,t

def spiral(n=1000,r=10,fiddle=.1,rmin=0,rmax=1,th0=0):
    theta = np.linspace(0,2*r*np.pi,n)+th0
    r = np.linspace(rmin,rmax,n)
    x = r*np.sin(theta)+np.random.normal(scale=fiddle,size=n)
    y = r*np.cos(theta)+np.random.normal(scale=fiddle,size=n)
    z = np.random.normal(size=n)

    return x,y,z

def mnms(mxr,word='PYTHON',mr=.15):
    cols = [
        'tab:red',
        'tab:green',
        'tab:blue',
        'gold',
        'saddlebrown',
        'tab:orange'
    ]        

    r = np.random.uniform(0.1,mxr*.8)
    t = np.random.uniform(0,2*np.pi)
    x,y = r*np.sin(t),r*np.cos(t)
    xs,ys = np.array([x]),np.array([y])
    while len(xs) < len(word):
        r = np.random.uniform(0.1,mxr*.93)
        t = np.random.uniform(0,2*np.pi)
        x,y = r*np.sin(t),r*np.cos(t)

        trad = np.sqrt((xs-x)*(xs-x) + (ys-y)*(ys-y))
        if trad.min() > mr*2.05:
            xs = np.concatenate((xs,[x]))
            ys = np.concatenate((ys,[y]))
        
        
    c = np.random.choice(cols,size=len(word))
    l = [_ for _ in word]
    np.random.shuffle(l)

    return xs,ys,c,list(l)

def one_cookie(ax,wrd='PYTHON'):
    n = np.random.randint(50,500)
    r = np.random.randint(3,7)
    f = np.random.uniform(.05,.1)
    
    sx,sy,sz = spiral(n=n,r=r,fiddle=f)
    mxr = max([sx.max(),sy.max()])
    bx,by1,by2,bt = border(mxr,n=35,fiddle=.015)

    pbx = np.concatenate(([-mxr],bx[bt],[mxr]))
    pby1 = np.concatenate(([0],by1[bt],[0]))
    pby2 = np.concatenate(([0],by2[bt],[0]))

    N = 250
    x = np.linspace(-1,1,N)
    X,Y = np.meshgrid(x,x)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            tw = g2d(sx,sy,X[i,j],Y[i,j])
            Z[i,j] = np.average(sz,weights=tw)

    colors = ['goldenrod','peru','chocolate','sienna']
    cmap = LSC.from_list('mycmap',colors)
        
    ax.imshow(Z,extent=[-1.1*mxr,1.1*mxr,-1.1*mxr,1.1*mxr],cmap=cmap)
    ax.fill_between(bx,1.1*mxr*np.ones(by1.shape),by1,color='w',zorder=1000)
    ax.fill_between(bx,-by2,-1.1*mxr*np.ones(by2.shape),color='w',zorder=1000)
    ax.plot(pbx,pby1,'k-',lw=4,zorder=1001)
    ax.plot(pbx,-pby2,'k-',lw=4,zorder=1001)

    mx,my,mc,ml = mnms(mxr,word=wrd)
    mR = 0.15
    
    props = {
        'ha': 'center',
        'va': 'center',
        'fontsize':12,
        'fontweight':'bold',
        'zorder':1007,
        'c':'w'
    }
        
    for x,y,c,l in zip(mx,my,mc,ml):
        circ = Circle((x,y),mR,color=c,zorder=1005,ec='k',lw=3)
        alph = [.3,.1] if c != 'gold' else [.5,.3]
        e1 = Ellipse((x+.38*mR,y+.38*mR),1.1*mR,.55*mR,-45,alpha=alph[0],color='w',zorder=1006,ec='None')
        e2 = Ellipse((x-.45*mR,y-.45*mR),mR,.45*mR,-45,alpha=alph[1],color='w',zorder=1006,ec='None')
        e3 = Ellipse((x-.35*mR,y-.35*mR),1.1*mR,.4*mR,-45,color=c,zorder=1007,ec='None')
        ax.add_patch(circ)
        ax.add_patch(e1)
        ax.add_patch(e2)
        ax.add_patch(e3)
        ax.text(x-.05*mR,y-.05*mR,l,props,rotation=np.random.uniform(0,360))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

F = plt.figure(figsize=(8,8),dpi=100)
words = [
    'PYTHON',
    'COOKIES',
    'MATPLOTLIB',
    'SWINBURNE'
]
for i in range(4):
    ax = F.add_subplot(2,2,i+1)
    one_cookie(ax,wrd=words[i])

plt.tight_layout()
plt.show()
