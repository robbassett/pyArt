import numpy as np

class SinuCirc(object):

    def __init__(self,radius,inner_omega):
        self.r = radius
        self.om= inner_omega
        
        self.x = np.linspace((-1.)*radius,radius,5000)
        self.t = np.arccos(self.x/radius)
        self.yi = np.sin(self.t)*radius
        self.yo = np.sin(self.t*inner_omega)*radius
        self.y = self.yi*self.yo/radius
        self.x1 = np.copy(self.x)
        self.y1 = np.copy(self.yi)
        self.y2 = np.copy(self.yo)

    def Spin(self,theta):

        rmat = np.array([[np.cos(theta),(-1.)*np.sin(theta)],
                               [np.sin(theta),np.cos(theta)]])
        xy   = np.vstack((self.x,self.y))
        self.x = np.sum(xy.T*rmat[0],axis=1)
        self.y = np.sum(xy.T*rmat[1],axis=1)
          




        
