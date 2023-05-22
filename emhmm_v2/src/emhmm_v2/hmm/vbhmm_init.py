import numpy as np
from sklearn.mixture import GaussianMixture

def vbhmm_init(datai, K, ini):
    VERBOSE_MODE = ini["verbose"]
    data = np.concatenate(datai, axis=0)
    N, dim = data.shape
    
    mix_t = {}
    mix_t["dim"] = dim
    mix_t["K"] = K
    mix_t["N"] = N
    
    # setup hyperparameters
    mix_t["alpha0"] = ini["alpha"]
    mix_t["epsilon0"] = ini["epsilon"]
    mix_t["m0"] = ini["mu"]
    mix_t["beta0"] = ini["beta"]
    
    if isinstance(ini["W"], float):
        mix_t["W0"] = ini["W"] * np.eye(dim)
    else:
        mix_t["W0"] = np.diag(ini["W"])
    
    if ini["v"] <= dim - 1:
        raise ValueError("v not large enough")
    mix_t["v0"] = ini["v"]
    mix_t["W0inv"] = np.linalg.inv(mix_t["W0"])
    
    switcher = {
        "random": initialize_random,
        "initgmm": initialize_with_given_gmm,
        "split": initialize_with_split
    }
    
    mix_t = switcher[ini["initmode"]](data, K, mix_t, ini)
    
    return mix_t


def initialize_random(data, K, mix_t, ini):
    try:
        gmm_opt = ini.get("random_gmm_opt", {})
        mix = GaussianMixture(n_components=K, **gmm_opt)
        mix.fit(data)
    except Exception:
        raise Exception("Failed to initialize GMM with random initialization.")
    
    mix_t["PComponents"] = mix.weights_
    mix_t["mu"] = mix.means_.T
    mix_t["Sigma"] = mix.covariances_
    
    return mix_t


def initialize_with_given_gmm(data, K, mix_t, ini):
    mix_t["Sigma"] = np.stack(ini["initgmm"]["cov"])
    mix_t["mu"] = np.concatenate(ini["initgmm"]["mean"], axis=0)
    
    if mix_t["mu"].shape[0] != K:
        raise ValueError("bad initgmm dimensions -- possibly mean is not a row vector")
    
    mix_t["PComponents"] = np.array(ini["initgmm"]["prior"])
    
    return mix_t


def initialize_with_split(data, K, mix_t, ini):
    gmm_opt = {
        "cvmode": "full",
        "initmode": "split",
        "verbose": 0
    }
    
    try:
        gmmmix = gmm_learn(data.T, K, gmm_opt)
    except Exception:
        raise Exception("Failed to initialize GMM with component splitting.")
    
    mix_t["PComponents"] = gmmmix["pi"]
    mix_t["mu"] = np.concatenate(gmmmix["mu"], axis=1).T
    mix_t["Sigma"] = np.stack(gmmmix["cv"])
    
    return mix_t
