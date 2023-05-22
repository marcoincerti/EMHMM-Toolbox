import numpy as np
from scipy.special import psi
from scipy.linalg import cholesky, det, solve_triangular
import vbhmm_init, vbhmm_fb

def vbhmm_em(data, K, ini):
    VERBOSE_MODE = ini['verbose']
    
    # get length of each chain
    trial = len(data)
    datalen = [d.shape[0] for d in data]
    lengthT = max(datalen)  # find the longest chain
    totalT = sum(datalen)
    
    # initialize the parameters
    mix_t = vbhmm_init(data, K, ini)  # initialize the parameters
    mix = mix_t
    dim = mix.dim  # dimension of the data
    K = mix.K  # no. of hidden states
    N = trial  # no. of chains
    maxT = lengthT  # the longest chain
    alpha0 = mix.alpha0  # hyper-parameter for the priors
    epsilon0 = mix.epsilon0  # hyper-parameter for the transitions
    m0 = mix.m0  # hyper-parameter for the mean
    beta0 = mix.beta0  # hyper-parameter for beta (Gamma)
    v0 = mix.v0  # hyper-parameter for v (Inverse-Wishart)
    W0inv = mix.W0inv  # hyper-parameter for Inverse-Wishart
    alpha = mix.alpha  # priors
    epsilon = mix.epsilon  # transitions
    beta = mix.beta  # beta (Gamma)
    v = mix.v  # v (Inverse-Wishart)
    m = mix.m  # mean
    W = mix.W  # Inverse-Wishart
    C = mix.C  # covariance
    const = mix.const  # constants
    const_denominator = mix.const_denominator  # constants
    maxIter = ini['maxIter']  # maximum iterations allowed
    minDiff = ini['minDiff']  # termination criterion
    
    L = -np.inf  # log-likelihood
    lastL = -np.inf  # log-likelihood
    
    # setup groups
    if ini['groups'] is not None:
        usegroups = 1
        group_ids = np.unique(ini['groups'])  # unique group ids
        numgroups = len(group_ids)
        group_inds = [np.where(ini['groups'] == g)[0] for g in group_ids]  # indices for group members
        
        # sanitized group membership (1 to G)
        group_map = np.zeros(len(ini['groups']))
        for g in range(numgroups):
            group_map[group_inds[g]] = g + 1
        
        # reshape alpha, epsilon into list
        # also Nk1 and M are lists
        epsilon = [epsilon] * numgroups
        alpha = [alpha] * numgroups
        
    else:
        usegroups = 0
    
    for iter in range(maxIter):
              
        ## E step
        
        if 1:
            ## 2016-08-14 - call function for FB algorithm
            
            # setup HMM
            fbhmm_varpar = {'v': v, 'W': W, 'epsilon': epsilon, 'alpha': alpha, 'm': m, 'beta': beta}
            fbopt = {'usegroups': usegroups}
            if usegroups:
                fbopt['group_map'] = group_map
                fbopt['numgroups'] = numgroups
            
            # call FB algorithm
            fbstats = vbhmm_fb(data, fbhmm_varpar, fbopt)
            
            # check stopping criteria
            if iter > 0:
                L = fbstats['loglik']
                diffL = (L - lastL) / abs(L)
                if VERBOSE_MODE:
                    print('iter %d, diffL = %f' % (iter, diffL))
                if diffL < minDiff:
                    break
                lastL = L
        
        if usegroups:
            for g in range(numgroups):
                # re-arrange group-specific stats
                groupT = sum([datalen[i] for i in group_inds[g]])
                group_fbstats = {'Nk': fbstats['Nk'][group_inds[g]], 'M': fbstats['M'][group_inds[g]],
                                 'T': groupT, 'loglik': fbstats['loglik'], 'lnXi': fbstats['lnXi'],
                                 'LnQ': fbstats['LnQ'][group_inds[g]]}
                
                # update parameters
                (alpha[g], epsilon[g], beta[g], v[g], m[g], W[g], C[g], const[g],
                 const_denominator[g]) = vbhmm_update(group_fbstats, alpha0, epsilon0,
                                                      beta0, v0, m0, W0inv)
                
                # compute group-specific constants
                const[g]['B'], const[g]['C'], const[g]['D'] = vbhmm_constants(v[g], C[g])
        
        else:
            # update parameters
            (alpha, epsilon, beta, v, m, W, C, const, const_denominator) = vbhmm_update(fbstats,
                                                                                       alpha0, epsilon0,
                                                                                       beta0, v0, m0, W0inv)
            
            # compute constants
            const['B'], const['C'], const['D'] = vbhmm_constants(v, C)
    
    return mix
