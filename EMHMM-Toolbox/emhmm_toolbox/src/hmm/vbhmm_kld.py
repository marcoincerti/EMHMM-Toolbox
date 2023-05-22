import numpy as np
import vbhmm_ll

def vbhmm_kld(hmm1, hmm2, data1, opt='n'):
    ll1 = vbhmm_ll(hmm1, data1, opt)
    ll2 = vbhmm_ll(hmm2, data1, opt)
    
    kld = np.mean(ll1 - ll2)
    
    return kld, ll1, ll2
