import imageio
import numpy as np

def mask_thing(fgim,bgim,mask_col):

    fgsz = fgim.shape
    bgsz = bgim.shape

    if bgsz[2] == 3:
        bgtm = np.zeros((bgsz[0],bgsz[1],4))+255
        bgtm[:,:,:3] = bgim
    bgim = np.copy(bgtm)

    nx = divmod(fgsz[0],bgsz[0])
    ny = divmod(fgsz[1],bgsz[1])

    
    if nx[0] == 0 and ny[0] == 0:
        bg = bgim[:nx[1],:ny[1],:]
    else:
        bg = np.zeros((bgsz[0]*(nx[0]+1),bgsz[1]*(ny[0]+1),4))
        for i in range(nx[0]+1):
            for j in range(ny[0]+1):
                bg[i*bgsz[0]:(i+1)*bgsz[0],j*bgsz[1]:(j+1)*bgsz[1],:] = bgim
        bg = bg[:fgsz[0],:fgsz[1],:]

    diff = np.abs(fgim-mask_col).sum(axis=2)
    r,c = np.where(diff < 50)

    out = np.copy(fgim)
    out[r,c,:] = bg[r,c,:]
    return out

if __name__ == '__main__':

    ff = imageio.imread('tmp.png')
    bb = imageio.imread('../random/egg.jpg')
    bs = bb.shape
    bg = np.zeros((bs[0],bs[1],4))+255
    bg[:,:,:3] = bb

    mc = np.array([255,0,0,255])
    tst = mask_thing(ff,bg,mc)

    imageio.imwrite('./taco.png',tst)
