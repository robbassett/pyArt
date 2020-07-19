import numpy as np
import matplotlib as mpl

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def sinuswitchcolor(length,periods,c1,c2):
    clt = np.linspace(0,2.*periods*np.pi,length)
    clt = (((-1.)*np.cos(clt))/2.)+0.5
    
    return [colorFader(c1,c2,_c) for _c in clt]
