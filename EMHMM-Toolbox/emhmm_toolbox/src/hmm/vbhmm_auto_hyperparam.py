import numpy as np
from PIL import Image

def vbhmm_auto_hyperparam(vbopt, data, img, opt):
    if vbopt is None:
        vbopt = {}
    
    if isinstance(img, str):
        img = Image.open(img)
    
    alldata = np.concatenate(data)
    if isinstance(alldata, list):
        alldata = np.concatenate(alldata)
    
    D = alldata.shape[1]
    
    if opt == 'c':
        vbopt['mu'] = 0.5 * np.array([img.shape[1], img.shape[0]])
        
        if D == 3:
            vbopt['mu'][2] = 250
        
        width = 0.5 * (img.shape[0] + img.shape[1])
        s = (width / 8) / 4
        
        if D == 2:
            vbopt['W'] = s**(-2)
        else:
            st = 25
            vbopt['W'] = np.array([s, s, st])**(-2)
    
    elif opt == 'd':
        vbopt['mu'] = np.mean(alldata, axis=0)
        
        vxy = np.var(alldata[:, :2], axis=0)
        s = np.sqrt(np.mean(vxy))
        
        if D == 2:
            vbopt['W'] = s**(-2)
        else:
            st = np.std(alldata[:, 2])
            vbopt['W'] = np.array([s, s, st])**(-2)
    
    else:
        raise ValueError('unknown option')
    
    return vbopt
