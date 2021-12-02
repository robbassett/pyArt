import imageio
import numpy as np

def fade_frames(im1,im2,images,n_fade):
    alpha = np.linspace(0,1,n_fade)
    for _a in alpha[:-1]:
        tm_im = (1.-_a)*im1 + (_a)*im2
        imageio.imwrite('frame.png',tm_im)
        images.append(imageio.imread('frame.png'))