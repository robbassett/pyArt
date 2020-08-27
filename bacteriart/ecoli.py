import numpy as np
import matplotlib.pyplot as plt

class Ecell():

    def __init__(self,
                 p,
                 v,
                 min_L=0.5,
                 max_L=1.,
                 W=0.2,
                 growth_rate = 0.1,
                 nspine = 1000
    ):
        
        self.t = 0.
        self.l = min_L
        self.p = p
        self.w = W
        self.v = v/np.sqrt(np.dot(v,v))
        self.ang = np.arctan(v[1]/v[0])
        self.nsp = nspine
        self.ml = max_L
        self.gr = growth_rate

        self.make_spine()

    def make_spine(self):
        spx,spy = np.cos(self.ang)*self.l/2.,np.sin(self.ang)*self.l/2.
        self.spine = np.array([np.linspace(self.p[0]-spx,self.p[0]+spx,self.nsp),
                                   np.linspace(self.p[1]-spy,self.p[1]+spy,self.nsp)])

    def grow(self):
        self.t += 1.
        self.l += (self.ml - self.l)*self.gr
        self.make_spine()
        
        

class Ecoliny():

    def __init__(self,max_L = 1.,W=0.2):

        pass


if __name__ == '__main__':

    ebuddy = Ecell([0.,0.],[1.,2.])
    F = plt.figure()
    ax = F.add_subplot(111)
    ax.plot(ebuddy.spine[0],ebuddy.spine[1],'k-')
    l = [ebuddy.l]
    for i in range(50):
        ebuddy.grow()
        ax.plot(ebuddy.spine[0]+i,ebuddy.spine[1],'k-')
        l.append(ebuddy.l)
    plt.show()

    F = plt.figure()
    ax = F.add_subplot(111)
    ax.plot(l,'ko')
    plt.show()
