import numpy as np
from scipy.stats import multivariate_normal


def vbhmm_ll(hmm, data, opt='n'):
    if opt != 'n' and opt != '':
        raise ValueError("Invalid option for 'opt'")
    
    if not isinstance(data, list):
        data = [data]
    
    prior = hmm['prior']
    transmat = hmm['trans']
    
    K = len(hmm['prior'])
    N = len(data)
    
    fb_qnorm = np.zeros(N)
    
    # for each sequence
    for n in range(N):
        tdata = data[n]
        tT = tdata.shape[0]
        
        # calculate observation likelihoods
        logrho = np.zeros((tT, K))
        for k in range(K):
            logrho[:, k] = np.log(multivariate_normal.pdf(tdata, mean=hmm['pdf'][k]['mean'], cov=hmm['pdf'][k]['cov']))
        
        # forward algorithm
        t_logPiTilde = prior.flatten()
        t_logATilde = transmat
        t_logrho = np.exp(logrho)
        
        t_alpha = np.zeros((tT, K))
        t_c = np.zeros(tT)
        
        if tT >= 1:
            # forward
            t_alpha[0, :] = t_logPiTilde * t_logrho[0, :]
            
            # rescale for numerical stability
            t_c[0] = np.sum(t_alpha[0, :])
            t_alpha[0, :] = t_alpha[0, :] / t_c[0]
            
            if tT > 1:
                for i in range(1, tT):
                    t_alpha[i, :] = np.dot(t_alpha[i - 1, :], t_logATilde) * t_logrho[i, :]
                    
                    # rescale for numerical stability
                    t_c[i] = np.sum(t_alpha[i, :])
                    t_alpha[i, :] = t_alpha[i, :] / t_c[i]
        
        # from scaling constants
        fb_qnorm[n] = np.sum(np.log(t_c))
    
    loglik = fb_qnorm
    
    # normalize
    if 'n' in opt:
        loglik = loglik / np.array([len(d) for d in data])
    
    errors = np.isinf(loglik)
    
    return loglik, errors
