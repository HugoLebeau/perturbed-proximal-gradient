import numpy as np

def mat2vec(mat):
    '''
    Transform a symmetric matrix into its vector representation.

    Parameters
    ----------
    mat : ndarray, shape (p, p)
        A symmetric matrix.

    Returns
    -------
    vec : ndarray, shape (n,)
        The vector representation.

    '''
    p = mat.shape[0]
    n = p*(p+1)//2
    vec = np.zeros(n)
    counter = 0
    for k in range(p):
        vec[counter:counter+p-k] = mat[k, k:]
        counter += p-k
    return vec

def vec2mat(vec):
    '''
    Transform a vectorized symmetric matrix into its matrix form.

    Parameters
    ----------
    vec : ndarray, shape (n,)
        A vectorized symmetric matrix.

    Returns
    -------
    mat : ndarray, shape (p, p)
        The corresponding matrix.

    '''
    n = vec.shape[0]
    p = np.int(np.round((-1+np.sqrt(1.+8.*n))/2.))
    mat = np.zeros((p, p))
    counter = 0
    for k in range(p):
        mat[k, k:] = vec[counter:counter+p-k]
        mat[k:, k] = vec[counter:counter+p-k]
        counter += p-k
    return mat
