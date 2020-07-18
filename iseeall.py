import numpy as np
import matplotlib.pyplot as plt

def draw_eye(ax,xy,rad,a,c1='k',c2='w'):

    eye_len = rad/2.
    xx = np.linspace(xy[0]-(eye_len/2.),xy[0]+(eye_len/2.),500)
    pup = 80.*eye_len
    lra = eye_len/1.7
    lr2 = eye_len

    ax.scatter([xy[0],xy[0],xy[0]],[xy[1],xy[1],xy[1]],s=[pup*4,pup*2,pup/2.],c=[c1,c2,c1],alpha=a,zorder=1000)
    lid_top = np.sqrt((lra*lra)-(xx*xx))
    lt2     = (-1.)*np.copy(lid_top)
    lid_top = lid_top-np.min(lid_top)+xy[1]
    lt2     = lt2-np.max(lt2)+xy[1]
    lid_bot = np.sqrt((lr2*lr2)-(xx*xx))
    lid_bot = lid_bot-np.min(lid_bot)+xy[1]
    ax.fill_between(xx,lid_top,lid_bot,color=c2,alpha=a,zorder=1001)
    ax.plot(xx,lid_top,'-',c=c1,lw=3,alpha=a,zorder=1002)
    ax.plot(xx,lid_bot,'-',c=c1,lw=2,alpha=a,zorder=1003)
    ax.plot(xx,lt2,'-',c=c1,lw=4,alpha=a,zorder=1004)

    tbot = xy[1]-eye_len/1.5
    bcx = eye_len/3.+(eye_len/2.)
    ttop = np.sin(np.pi/3.)*(bcx)+xy[1]
    tx = [(-1.)*bcx+xy[0],0,bcx+xy[0],(-1.)*bcx+xy[0]]
    ty = np.array([tbot,ttop,tbot,tbot])+(np.sin(np.pi/3.)*(eye_len/4.))
    ax.fill_between(tx[:-1],ty[:-1],[ty[0],ty[0],ty[0]],color=c2,zorder=998,alpha=a)
    ax.plot(tx,ty,'-',c=c1,lw=4,alpha=a,zorder=999)

if __name__ == '__main__':
    f=plt.figure()
    ax=f.add_subplot(111)
    draw_eye(ax,[0,-1.5],5.,.5)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_aspect('equal')
    plt.show()

    xx=np.linspace(0,50,500)
    yy=np.exp((-1.*xx))
    xx=np.linspace(0,50,1000)
    yy=np.concatenate((np.flip(yy),yy))
    F =plt.figure()
    ax = F.add_subplot(111)
    ax.plot(xx,yy,'k-')
    plt.show()
