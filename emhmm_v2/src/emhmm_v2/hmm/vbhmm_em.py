import numpy as np
import emhmm_v2.hmm.vbhmm_init as vbhmm_init
import emhmm_v2.hmm.vbhmm_fb as vbhmm_fb

def vbhmm_em(data, K, ini):
    VERBOSE_MODE = ini['verbose']
    
    # Get length of each chain
    trial = len(data)
    datalen = [np.array(d).shape[0] for d in data]
    lengthT = max(datalen)
    totalT = sum(datalen)
    
    # Initialize the parameters
    mix_t = vbhmm_init(data, K, ini)
    mix = mix_t
    dim = mix.dim
    K = mix.K
    N = trial
    maxT = lengthT
    alpha0, epsilon0, m0, beta0, v0, W0inv = mix.alpha0, mix.epsilon0, mix.m0, mix.beta0, mix.v0, mix.W0inv
    alpha, epsilon, beta, v, m, W, C, const, const_denominator = mix.alpha, mix.epsilon, mix.beta, mix.v, mix.m, mix.W, mix.C, mix.const, mix.const_denominator
    maxIter, minDiff = ini['maxIter'], ini['minDiff']
    
    L = -np.inf
    lastL = -np.inf
    
    # Setup groups
    if ini['groups'] is not None:
        usegroups = 1
        group_ids = np.unique(ini['groups'])
        numgroups = len(group_ids)
        group_inds = [np.where(ini['groups'] == g)[0] for g in group_ids]
        group_map = np.zeros(len(ini['groups']))
        for g in range(numgroups):
            group_map[group_inds[g]] = g + 1
        epsilon = [epsilon] * numgroups
        alpha = [alpha] * numgroups
    else:
        usegroups = 0
    
    for iter in range(maxIter):
              
        ## E step
        
        if 1:
            fbhmm_varpar = {'v': v, 'W': W, 'epsilon': epsilon, 'alpha': alpha, 'm': m, 'beta': beta}
            fbopt = {'usegroups': usegroups}
            if usegroups:
                fbopt['group_map'] = group_map
                fbopt['numgroups'] = numgroups
            fbstats = vbhmm_fb(data, fbhmm_varpar, fbopt)
            
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
                groupT = sum([datalen[i] for i in group_inds[g]])
                group_fbstats = {'Nk': fbstats['Nk'][group_inds[g]], 'M': fbstats['M'][group_inds[g]],
                                 'T': groupT, 'loglik': fbstats['loglik'], 'lnXi': fbstats['lnXi'],
                                 'LnQ': fbstats['LnQ'][group_inds[g]]}
                
                (alpha[g], epsilon[g], beta[g], v[g], m[g], W[g], C[g], const[g],
                 const_denominator[g]) = vbhmm_update(group_fbstats, alpha0, epsilon0,
                                                      beta0, v0, m0, W0inv)
                
                const[g]['B'], const[g]['C'], const[g]['D'] = vbhmm_constants(v[g], C[g])
        
        else:
            (alpha, epsilon, beta, v, m, W, C, const, const_denominator) = vbhmm_update(fbstats,
                                                                                       alpha0, epsilon0,
                                                                                       beta0, v0, m0, W0inv)
            
            const['B'], const['C'], const['D'] = vbhmm_constants(v, C)
    
    return mix
