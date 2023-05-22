import numpy as np
import vbhmm_permute

def vbhmm_standardize(hmm, mode):
    """
    vbhmm_standardize - standardize an HMM's states (ROIs) to be consistent.

    Args:
        hmm (dict): An HMM from vbhmm_learn or a group HMM from vhem_cluster.
        mode (str): 'e' - sort by emission frequency (overall number of fixations in an ROI)
                    'p' - sort by prior frequency (number of first-fixations in an ROI)
                    'f' - sort by most-likely fixation path
                          (state 1 is most likely first fixation. State 2 is most likely 2nd fixation, etc)

    Returns:
        dict: The standardized HMM.

    ---
    Eye-Movement analysis with HMMs (emhmm-toolbox)
    Copyright (c) 2017-01-13
    Antoni B. Chan, Janet H. Hsiao, Tim Chuk
    City University of Hong Kong, University of Hong Kong

    2016-11-22: ABC - created
    """

    # run on each hmm
    if 'hmms' in hmm:
        hmm_new = hmm.copy()
        hmm_new['hmms'] = [vbhmm_standardize(sub_hmm, mode) for sub_hmm in hmm['hmms']]
        return hmm_new

    if mode in ['d', 'e']:
        if mode == 'd':
            print("Warning: standardization mode 'd' is deprecated. Use 'e'")
        
        if 'N' in hmm:
            wsort = np.sort(hmm['N'], axis=0)[::-1]
            wi = np.argsort(hmm['N'], axis=0)[::-1]
        else:
            raise ValueError("Cluster size unknown - not from vbhmm_learn")

    elif mode == 'p':
        wsort = np.sort(hmm['prior'], axis=None)[::-1]
        wi = np.argsort(hmm['prior'], axis=None)[::-1]

    elif mode == 'f':
        A = hmm['trans']
        wi = np.zeros(len(hmm['prior']), dtype=int)
        for t in range(len(hmm['prior'])):
            if t == 0:
                curf = np.argmax(hmm['prior'])
            else:
                curf = np.argmax(A[curf, :])
            wi[t] = curf
            A[:, curf] = -1

    else:
        raise ValueError("Unknown mode")

    # permute states
    hmm_new = vbhmm_permute(hmm, wi)
    return hmm_new
