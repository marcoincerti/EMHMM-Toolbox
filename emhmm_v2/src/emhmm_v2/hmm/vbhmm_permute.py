def vbhmm_permute(hmm_old, cl):
    """
    vbhmm_permute - permute the state indices of an HMM

    Args:
        hmm_old (dict): The original HMM.
        cl (list): The permutation mapping from ROI "cl(i)" in hmm_old to ROI "i" in hmm.

    Returns:
        dict: The permuted HMM.

    ---
    Eye-Movement analysis with HMMs (emhmm-toolbox)
    Copyright (c) 2017-01-13
    Antoni B. Chan, Janet H. Hsiao, Tim Chuk
    City University of Hong Kong, University of Hong Kong
    """

    hmm = hmm_old.copy()

    usegroups = 'group_ids' in hmm

    if not usegroups:
        # without groups
        hmm['prior'] = hmm_old['prior'][cl]
        hmm['trans'] = hmm_old['trans'][cl][:, cl]
        if 'M' in hmm_old:
            hmm['M'] = hmm_old['M'][cl][:, cl]
        if 'N1' in hmm_old:
            hmm['N1'] = hmm_old['N1'][cl]
    else:
        # with groups
        for g in range(len(hmm['prior'])):
            hmm['prior'][g] = hmm_old['prior'][g][cl]
            hmm['trans'][g] = hmm_old['trans'][g][cl][:, cl]
            hmm['M'][g] = hmm_old['M'][g][cl][:, cl]
            hmm['N1'][g] = hmm_old['N1'][g][cl]
            hmm['Ng'][g] = hmm_old['Ng'][g][cl]

    if 'N' in hmm_old:
        hmm['N'] = hmm_old['N'][cl]
    hmm['pdf'] = [hmm_old['pdf'][i] for i in cl]

    if 'gamma' in hmm_old:
        for n in range(len(hmm['gamma'])):
            hmm['gamma'][n] = hmm_old['gamma'][n][cl, :]

    if 'varpar' in hmm:
        if not usegroups:
            hmm['varpar']['epsilon'] = hmm_old['varpar']['epsilon'][cl][:, cl]
            hmm['varpar']['alpha'] = hmm_old['varpar']['alpha'][cl]
        else:
            for g in range(len(hmm['prior'])):
                hmm['varpar']['epsilon'][g] = hmm_old['varpar']['epsilon'][g][cl][:, cl]
                hmm['varpar']['alpha'][g] = hmm_old['varpar']['alpha'][g][cl]
        hmm['varpar']['beta'] = hmm_old['varpar']['beta'][cl]
        hmm['varpar']['v'] = hmm_old['varpar']['v'][cl]
        hmm['varpar']['m'] = hmm_old['varpar']['m'][:, cl]
        hmm['varpar']['W'] = hmm_old['varpar']['W'][:, :, cl]

    return hmm
