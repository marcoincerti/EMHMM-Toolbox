import numpy as np
from emhmm_v2.hmm.vbhmm_em import vbhmm_em
from scipy.stats import multivariate_normal
from scipy.special import digamma
from sklearn.mixture import GaussianMixture

def vbhmm_learn(data, K, vbopt=None):
    if vbopt is None:
        vbopt = {}
    
    vbopt.setdefault('alpha', 0.1)
    
    if 'mean' in vbopt:
        print('DEPRECATED: vbopt.mean has been renamed to vbopt.mu')
        vbopt['mu'] = vbopt['mean']
    
    D = np.array(data[0]).shape[1]
    if D == 2:
        defmu = np.array([256, 192])
    elif D == 3:
        defmu = np.array([256, 192, 150])
    else:
        raise ValueError(f'No default mu for D={D}')
    
    vbopt.setdefault('mu', defmu)
    vbopt.setdefault('W', 0.005)
    vbopt.setdefault('beta', 1)
    vbopt.setdefault('v', 5)
    vbopt.setdefault('epsilon', 0.1)
    vbopt.setdefault('initmode', 'random')
    vbopt.setdefault('numtrials', 50)
    vbopt.setdefault('maxIter', 100)
    vbopt.setdefault('minDiff', 1e-5)
    vbopt.setdefault('showplot', 1)
    vbopt.setdefault('sortclusters', 'f')
    vbopt.setdefault('groups', [])
    vbopt.setdefault('fix_clusters', 0)
    vbopt.setdefault('random_gmm_opt', {})
    vbopt.setdefault('fix_cov', [])
    vbopt.setdefault('verbose', 1)
    
    VERBOSE_MODE = vbopt['verbose']
    
    if isinstance(K, list):
        vbopt2 = vbopt.copy()
        vbopt2['showplot'] = 0
        
        out_all = []
        LLk_all = np.zeros(len(K))
        for ki in range(len(K)):
            if VERBOSE_MODE >= 2:
                print(f'-- K={K[ki]} --')
            elif VERBOSE_MODE == 1:
                print(f'-- vbhmm K={K[ki]}: ', end='')
            
            if vbopt2['initmode'] == 'initgmm':
                vbopt2['initgmm'] = vbopt['initgmm'][ki]
            
            out_all.append(vbhmm_learn(data, K[ki], vbopt2))
            LLk_all[ki] = out_all[ki]['LL']
        
        LLk_all += np.log(np.arange(1, len(K)+1)).sum()
        maxLLk = np.max(LLk_all)
        ind = np.argmax(LLk_all)
        
        hmm = out_all[ind]
        hmm['model_LL'] = LLk_all
        hmm['model_k'] = K
        hmm['model_bestK'] = K[ind]
        L = maxLLk
        
        if VERBOSE_MODE >= 1:
            print(f'best model: K={K[ind]}; L={maxLLk}')
        
        if vbopt['showplot']:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(K, LLk_all, 'b.-')
            plt.plot([min(K), max(K)], [maxLLk, maxLLk], 'k--')
            plt.plot(K[ind], maxLLk, 'bo')
            plt.grid(True)
            plt.xlabel('K')
            plt.ylabel('data log likelihood')
            plt.show()
    
    else:
        if vbopt['initmode'] == 'random':
            numits = vbopt['numtrials']
            vb_hmms = []
            LLall = np.zeros(numits)
            for it in range(numits):
                if VERBOSE_MODE == 1:
                    print(f'{it} ', end='')
                
                vb_hmms.append(vbhmm_em(data, K, vbopt))
                LLall[it] = vb_hmms[it]['LL']
            
            maxLL = np.max(LLall)
            maxind = np.argmax(LLall)
            
            if VERBOSE_MODE >= 1:
                print(f'\nbest run={maxind}; LL={maxLL}')
            
            hmm = vb_hmms[maxind]
            L = hmm['LL']
        
        elif vbopt['initmode'] == 'initgmm':
            hmm = vbhmm_em(data, K, vbopt)
            L = hmm['LL']
        
        elif vbopt['initmode'] == 'split':
            hmm = vbhmm_em(data, K, vbopt)
            L = hmm['LL']
    
    if vbopt['sortclusters']:
        hmm_old = hmm.copy()
        hmm = vbhmm_standardize(hmm, vbopt['sortclusters'])
    
    if vbopt['showplot']:
        vbhmm_plot(hmm, data)
    
    hmm['vbopt'] = vbopt
    
    return hmm, L

def setdefault(vbopt, field, value):
    if field not in vbopt:
        vbopt[field] = value
    return vbopt
