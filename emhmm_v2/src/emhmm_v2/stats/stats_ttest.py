import numpy as np
from scipy.stats import ttest_ind
import vbhmm_ll

def stats_ttest(hmms1, hmms2, data1, opt='n'):
    N = len(data1)
    allLL1 = np.zeros(N)
    allLL2 = np.zeros(N)
    
    for i in range(N):
        if isinstance(hmms1, list):
            myhmm1 = hmms1[i]
        else:
            myhmm1 = hmms1
        
        if isinstance(hmms2, list):
            myhmm2 = hmms2[i]
        else:
            myhmm2 = hmms2
        
        ll1 = vbhmm_ll(myhmm1, data1[i], opt)
        ll2 = vbhmm_ll(myhmm2, data1[i], opt)
        
        allLL1[i] = np.mean(ll1)
        allLL2[i] = np.mean(ll2)
    
    _, p, ci, stats = ttest_ind(allLL1, allLL2, alternative='greater')
    
    lld = allLL1 - allLL2
    info = stats
    info['ci'] = ci
    
    return p, info, lld
