import numpy as np


def gmm_ll(X, gmm):
    K = gmm['K']
    d = X.shape[0]
    N = X.shape[1]

    if K > 200 and nargout == 1:
        # if only looking for LL, and K large
        # calculate LL only
        LLmax = np.zeros(N)
        for c in range(K):
            tmpLL = logComponent(gmm, X, c, d, N)
            LLmax = np.maximum(LLmax, tmpLL)

        if 'bkgndclass' in gmm and gmm['bkgndclass'] != 0:
            raise ValueError('not supported')

        # do logtrick
        cterm = np.zeros(N)
        for c in range(K):
            tmpLL = logComponent(gmm, X, c, d, N)
            cterm = cterm + np.exp(tmpLL - LLmax)

        LL = LLmax + np.log(cterm)

        # sanity check
        if 0:
            LLcomp = np.zeros((N, K))
            for c in range(K):
                LLcomp[:, c] = logComponent(gmm, X, c, d, N)
            LL2 = logtrick2(LLcomp)
            totalerror(LL, LL2)

    else:
        # calculate LL and LLcomp
        # component log-likelihood
        LLcomp = np.zeros((N, K))

        # for each class
        for c in range(K):
            LLcomp[:, c] = logComponent(gmm, X, c, d, N)

        # add background class
        if 'bkgndclass' in gmm and gmm['bkgndclass'] != 0:
            LLcomp = np.hstack((LLcomp, np.log(gmm['bkgndclass'])))

        LL = logtrick2(LLcomp)

    # output posterior
    if nargout > 2:
        post = np.exp(LLcomp - np.tile(LL, (1, K)))
        return LL, LLcomp, post
    else:
        return LL


# return the log-likelihood of component c
def logComponent(gmm, X, c, d, N):
    # initialize
    myLLcomp = np.zeros(N)

    # setup pdf constants
    mu = gmm['mu'][c]
    cv = gmm['cv'][c]

    tempx = X - np.tile(mu, (N, 1)).T

    if gmm['cvmode'] == 'iid':
        g = np.sum(tempx * tempx, axis=0) / cv
        ld = d * np.log(cv)

    elif gmm['cvmode'] == 'diag':
        # g = np.sum((np.tile(1 / cv, (1, N)) * tempx) * tempx, axis=0)
        # This is not as numerically stable, but much faster
        # g = np.sum(tempx * tempx * np.tile(1 / cv, (1, N)), axis=0)
        # This is more numerically stable.
        tmp = tempx * np.tile(1 / np.sqrt(cv), (1, N))
        g = np.sum(tmp * tmp, axis=0)
        ld = np.sum(np.log(cv))

    elif gmm['cvmode'] == 'full':
        g = np.sum(np.dot(np.linalg.inv(cv), tempx) * tempx, axis=0)
        ld = np.log(np.linalg.det(cv))

    else:
        raise ValueError('bad mode')

    myLLcomp = -0.5 * g - (d / 2) * np.log(2 * np.pi) - 0.5 * ld + np.log(gmm['pi'][c])

    return myLLcomp


def logtrick2(lA):
    # logtrick - "log sum trick" - calculate log(sum(A)) using only log(A)
    mv = np.max(lA, axis=1)
    temp = lA - np.tile(mv, (lA.shape[1], 1)).T
    cterm = np.sum(np.exp(temp), axis=1)
    s = mv + np.log(cterm)
    return s


def logtrick(lA):
    # logtrick - "log sum trick" - calculate log(sum(A)) using only log(A)
    mv = np.max(lA, axis=0)
    temp = lA - np.tile(mv, (lA.shape[0], 1))
    cterm = np.sum(np.exp(temp), axis=0)
    s = mv + np.log(cterm)
    return s


def logdet(A):
    # R'*R = A, R is triangular
    # det(R'*R) = det(A) = det(R)^2

    # R = np.linalg.cholesky(A)
    # ld = 2 * np.sum(np.log(np.diag(R)))

    R, p = np.linalg.cholesky(A), np.linalg.cholesky(A)
    if p == 0:
        ld = 2 * np.sum(np.log(np.diag(R)))
    else:
        x = np.linalg.eigvals(A)
        ld = np.sum(np.log(x))
        warnings.warn('logdet:chol', 'A is not PD for chol, using eig')

    return ld
