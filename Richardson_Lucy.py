import numpy as np
import scipy.signal
    
###############################################################################
def convolve(im1, psf_p):
    """ Convolves an image with a PSF """
    im2 = scipy.signal.convolve(im1, psf_p, mode='same')
    im2 [im2 < 0] = 0
    im2[im2 < 0.000001 * np.max(psf_p)] = 0
    return im2

###############################################################################
def RL(updated_image, original_image, psf):
    """ Carries out Richardson-Lucy convolution"""
    if len(updated_image.shape) == 2:
        rl_image = scipy.signal.convolve2d(updated_image, psf, mode='same')
        rl_image[rl_image==0] = 1
        rl_image = original_image / rl_image
        rl_image = scipy.signal.convolve2d(rl_image, psf, mode='same')
        
    if len(updated_image.shape) == 3:
        rl_image = scipy.signal.convolve(updated_image, psf, mode='same')
        rl_image[rl_image<=0] = 1
        rl_image = original_image / rl_image
        rl_image = scipy.signal.convolve(rl_image, psf, mode='same')
    
    rl_image[rl_image < 0] = 0
    rl_image[rl_image < 0.001*np.nanmax(rl_image)] = 0    
        
    return rl_image

###############################################################################
def noise_regularisation(updated_image, lamb, eng):
    """ Noise regularisation for RL"""
    
    scipy.io.savemat('C:\\Users\\cg12\\Documents\\Python Scripts\\saved_variables\\PVC_RL_in.mat', {'image':updated_image})
    eng.PVC_NR(nargout=0)
    div = scipy.io.loadmat('C:\\Users\\cg12\Documents\\Python Scripts\\saved_variables\\PVC_RL_out.mat')
    div = div['div']
    nr = 1 / (1 - (lamb * div))
    nr = np.nan_to_num(nr)
    nr[nr == 0] = 1
    
    return nr
###############################################################################
def RL_NR(updated_image, original_image, psf_p, lamb):
    """ RL with noise regularisation"""
    
    def divergence(f):
        """ Divergence of an array"""
        num_dims = len(f)
        return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])
    
    """Calculate RL component """
    rl_image = scipy.signal.convolve(updated_image, psf_p, mode='same')
    rl_image[rl_image<=0] = 1
    rl_image = original_image / rl_image
    rl_image = scipy.signal.convolve(rl_image, psf_p, mode='same')
    rl_image[rl_image < 0] = 0

    """Calculate NR component """
    grad = np.gradient(updated_image)
    h = np.zeros_like(grad[0])
    for gradi in range(len(grad)):
        h += grad[gradi] * grad[gradi]
    h = np.sqrt(h)
    h[h == 0]=1
    div = divergence(grad / h)
    div = 1 - (lamb * div)
    nr = np.nan_to_num(div ** -1)
    
    """combine for updated image """
    output = updated_image * rl_image * nr
    
    return output
############################################################################### 
def apply_RL_NR(im, psf_p, n, lamb):
    """ Apply n iterations 0f RL-NR"""
    global eng
    updated_image = im.copy()
    original_image = im.copy()
    for ii in range(n):
        updated_image = RL_NR(updated_image, original_image, psf_p, lamb)
    return updated_image
############################################################################### 
    