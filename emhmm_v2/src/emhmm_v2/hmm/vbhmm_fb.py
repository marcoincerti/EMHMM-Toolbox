import numpy as np

def vbhmm_fb(data, hmm_varpar, opt):
    # HMM parameters: v, W, epsilon, alpha, m, beta
    v = hmm_varpar['v']
    W = hmm_varpar['W']
    epsilon = hmm_varpar['epsilon']
    alpha = hmm_varpar['alpha']
    m = hmm_varpar['m']
    beta = hmm_varpar['beta']
    
    # Options: usegroups
    usegroups = opt['usegroups']
    if usegroups:
        group_map = opt['group_map']
        numgroups = opt['numgroups']
    
    # Other options: savexi
    savexi = opt.get('savexi', 0)
    
    # Constants: K, dim, N, maxT, lengthT, const, const_denominator
    K = m.shape[1]
    dim = m.shape[0]
    N = len(data)
    maxT = max([np.prod(data[i].shape) / dim for i in range(N)])
    lengthT = maxT
    const = dim * np.log(2)
    const_denominator = (dim * np.log(2 * np.pi)) / 2
    
    # Pre-calculate constants
    logLambdaTilde = np.zeros(K)
    for k in range(K):
        t1 = np.psi(0, 0.5 * (v[k] + 1) - 0.5 * np.arange(1, dim+1))
        logLambdaTilde[k] = np.sum(t1) + const + np.log(np.linalg.det(W[:,:,k]))
    
    if not usegroups:
        psiEpsilonHat = np.zeros(K)
        logATilde = np.zeros((K, K))
        for k in range(K):
            psiEpsilonHat[k] = np.psi(0, np.sum(epsilon[:,k]))
            logATilde[:,k] = np.psi(0, epsilon[:,k]) - psiEpsilonHat[k]
        psiAlphaHat = np.psi(0, np.sum(alpha))
        logPiTilde = np.psi(0, alpha) - psiAlphaHat
    else:
        psiEpsilonHat = [np.zeros(K) for _ in range(numgroups)]
        logATilde = [np.zeros((K, K)) for _ in range(numgroups)]
        psiAlphaHat = [np.zeros(K) for _ in range(numgroups)]
        logPiTilde = [np.zeros(K) for _ in range(numgroups)]
        for g in range(numgroups):
            for k in range(K):
                psiEpsilonHat[g][k] = np.psi(0, np.sum(epsilon[g][:,k]))
                logATilde[g][:,k] = np.psi(0, epsilon[g][:,k]) - psiEpsilonHat[g][k]
            psiAlphaHat[g] = np.psi(0, np.sum(alpha[g]))
            logPiTilde[g] = np.psi(0, alpha[g]) - psiAlphaHat[g]
    
    logrho_Saved = np.zeros((K, N, lengthT))
    fb_qnorm = np.zeros(N)
    
    gamma_sum = np.zeros((K, N, maxT))
    sumxi_sum = np.zeros((K, K, N))
    
    if savexi:
        xi_Saved = [np.zeros((K, K, data[n].shape[0])) for n in range(N)]
    
    for n in range(N):
        tdata = data[n]
        T = np.prod(tdata.shape) / dim
        
        logrho = np.zeros((K, T))
        xi = np.zeros((K, K, T-1))
        
        if not usegroups:
            psiQhat = np.zeros(K)
            E_z = np.zeros(K)
            gamma = np.zeros((K, T))
            sumxi = np.zeros((K, K))
            
            for k in range(K):
                psiQhat[k] = np.psi(0, beta[k] + T * np.sum(gamma[k,:]))
                E_z[k] = (psiQhat[k] - np.log(T) - const_denominator
                          - 0.5 * np.sum(np.sum(gamma[k,:]) * logLambdaTilde))
                gamma[k,0] = E_z[k] + 0.5 * np.sum(logLambdaTilde)
            
            for t in range(T-1):
                gamma_sum[:,n,t] = gamma[:,t]
                
                logrho[:,t] = (const_denominator - 0.5 * dim * (1 / beta)
                               + 0.5 * np.dot(np.dot(v, np.dot(W, tdata[t+1])),
                                              tdata[t+1]))
                
                temp = (logrho[:,t] + gamma[:,t]).reshape(K, 1) + logATilde
                Amax = np.max(temp, axis=0)
                Asum = np.sum(np.exp(temp - Amax), axis=0)
                logAtmp = Amax + np.log(Asum)
                
                xi[:,:,t] = np.exp(temp - Amax - logAtmp)
                
                gamma[:,t+1] = E_z + np.sum(logAtmp.reshape(K, 1)
                                            + logrho[:,t].reshape(K, 1) + gamma[:,t].reshape(K, 1),
                                            axis=0)
                
                sumxi = sumxi + xi[:,:,t]
        
            gamma_sum[:,n,T-1] = gamma[:,T-1]
            
            if savexi:
                xi_Saved[n] = xi
        
        else:
            group_idx = group_map[n]
            psiQhat = np.zeros(K)
            E_z = np.zeros(K)
            gamma = np.zeros((K, T))
            sumxi = np.zeros((K, K))
            
            for k in range(K):
                psiQhat[k] = np.psi(0, beta[group_idx][k] + T * np.sum(gamma[k,:]))
                E_z[k] = (psiQhat[k] - np.log(T) - const_denominator
                          - 0.5 * np.sum(np.sum(gamma[k,:]) * logLambdaTilde))
                gamma[k,0] = E_z[k] + 0.5 * np.sum(logLambdaTilde)
            
            for t in range(T-1):
                gamma_sum[:,n,t] = gamma[:,t]
                
                logrho[:,t] = (const_denominator - 0.5 * dim * (1 / beta[group_idx])
                               + 0.5 * np.dot(np.dot(v[group_idx], np.dot(W[group_idx], tdata[t+1])),
                                              tdata[t+1]))
                
                temp = (logrho[:,t] + gamma[:,t]).reshape(K, 1) + logATilde[group_idx]
                Amax = np.max(temp, axis=0)
                Asum = np.sum(np.exp(temp - Amax), axis=0)
                logAtmp = Amax + np.log(Asum)
                
                xi[:,:,t] = np.exp(temp - Amax - logAtmp)
                
                gamma[:,t+1] = E_z + np.sum(logAtmp.reshape(K, 1)
                                            + logrho[:,t].reshape(K, 1) + gamma[:,t].reshape(K, 1),
                                            axis=0)
                
                sumxi = sumxi + xi[:,:,t]
        
            gamma_sum[:,n,T-1] = gamma[:,T-1]
            
            if savexi:
                xi_Saved[n] = xi
        
        if usegroups:
            logrho_Saved[:,n,:T] = logrho[:, :T]
        
        fb_qnorm[n] = np.max(gamma[:,0])
        
        sumxi_sum[:,:,n] = sumxi
    
    if usegroups:
        sumxi = np.zeros((K, K))
        gamma = np.zeros((K, maxT))
        
        for g in range(numgroups):
            group_indices = [i for i in range(N) if group_map[i] == g]
            group_lengths = [data[i].shape[0] for i in group_indices]
            group_maxT = max(group_lengths)
            
            sumxi_group = np.zeros((K, K))
            sumgamma_group = np.zeros(K)
            
            for i, n in enumerate(group_indices):
                sumxi_group += np.pad(xi_Saved[n][:,:, :group_lengths[i]-1],
                                      ((0,0), (0,0), (0,group_maxT-group_lengths[i]+1)),
                                      mode='constant')
                sumgamma_group += np.pad(gamma_sum[:,n, :group_lengths[i]],
                                         (0, group_maxT-group_lengths[i]),
                                         mode='constant')
            
            sumxi += sumxi_group
            gamma += np.pad(sumgamma_group.reshape(K, 1),
                            (0, maxT-group_maxT),
                            mode='constant')
        
        for k in range(K):
            psiQhat = np.psi(0, np.sum(sumgamma_group) + beta[k])
            E_z[k] = (psiQhat - np.log(maxT) - const_denominator
                      - 0.5 * np.sum(sumgamma_group * logLambdaTilde))
            gamma[k,0] = E_z[k] + 0.5 * np.sum(logLambdaTilde)
        
        for t in range(maxT-1):
            logrho = (const_denominator - 0.5 * dim * (1 / beta)
                      + 0.5 * np.dot(np.dot(v, np.dot(W, data[0][t+1])),
                                      data[0][t+1]))
            
            temp = (logrho + gamma[:,t]).reshape(K, 1) + logATilde
            Amax = np.max(temp, axis=0)
            Asum = np.sum(np.exp(temp - Amax), axis=0)
            logAtmp = Amax + np.log(Asum)
            
            xi[:,:,t] = np.exp(temp - Amax - logAtmp)
            
            gamma[:,t+1] = E_z + np.sum(logAtmp.reshape(K, 1)
                                        + logrho.reshape(K, 1) + gamma[:,t].reshape(K, 1),
                                        axis=0)
        
        sumxi_sum[:,:,0] = sumxi
        gamma_sum[:,:,0] = gamma
    
    if not usegroups:
        sumxi_sum = sumxi_sum[:,:,0]
        gamma_sum = gamma_sum[:,:,0]
        logrho_Saved = logrho_Saved[:,:,0]
    
    return sumxi_sum, gamma_sum, logrho_Saved, xi_Saved, fb_qnorm
