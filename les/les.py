import numpy as np
import scipy.spatial as spat
import scipy.linalg as spla
from scipy.sparse.linalg import eigsh


APPROXEIG = True    # Use [Tropp et al., 2017] approximation for the eigenvalues
M = 2               # Number of random vectors to use in [Tropp et al., 2017] (total = M x nev)


def _build_graph(data, sigma=2):
    """
    Compute an approximation of the heat kernel based on diffusion maps

    :param data: data samples organized as [samples x features]
    :param sigma: kernel scale (multiplying the median of the distances)
    :return h_op: a Discrete approximation of the heat kernel using diffusion maps
    """

    # -------------- Distance matrix and kernel computation: -------------
    dist_mat = np.square(spat.distance.squareform(spat.distance.pdist(data)))
    dist_mat = np.exp(-dist_mat / (sigma * np.median(dist_mat)))

    # ------------ Construction of the symmetric diffusion operator: ------------
    h_op = dist_mat
    d = 1 / np.sum(dist_mat, axis=1)
    for i in range(h_op.shape[0]):
        h_op[i, :] *= d[i]
        h_op[:, i] *= d[i]

    d2 = 1 / np.sqrt(np.sum(h_op, axis=1))
    for i in range(h_op.shape[0]):
        h_op[i, :] *= d2[i]
        h_op[:, i] *= d2[i]

    return h_op


def _compute_log_eigenvalues(h_op, nev=500, gamma=1e-6, tol=1e-8):
    """
    Estimating the eigenvalues

    :param h_op: discrete approximation of the heat kernel using diffusion maps (PSD matrix)
    :param nev: number of eigenvalues to compute
    :param gamma: kernel regularization parameter
    :param tol: tolerance for eigenvalue computation if not using the approximation
    :return levals: log of the estimated eigenvalues (with regularization parameter)
    """

    if not APPROXEIG:

        eigvals = eigsh(h_op, k=nev, return_eigenvectors=False, tol=tol, sigma=1, which='LM')

    else:
        # Fixed rank PSD approximation algorithm [Tropp et al., 2017]
        mu = 2.2 * 1e-16
        n = h_op.shape[0]

        omega = np.random.randn(n, M * nev)
        omega = spla.orth(omega)

        y = h_op @ omega
        nu = mu * np.linalg.norm(y, ord=2)

        y_nu = y + nu * omega
        b_mat = omega.T @ y_nu
        c_mat = np.linalg.cholesky((b_mat + b_mat.T) / 2).T
        eigvals = spla.svdvals(y_nu @ np.linalg.inv(c_mat))
        eigvals = np.maximum(np.square(eigvals) - nu, 0)
        eigvals = np.sort(eigvals)[-nev:]

    log_eigvals = np.log(eigvals + gamma)

    return log_eigvals, eigvals


def les_desc_comp(data, sigma=2, nev=500, gamma=1e-6):
    """
    Compute LES descriptors

    :param data: data samples organized as [samples x features]
    :param sigma: kernel scale for diffusion operator (multiplying the median of the distances)
    :param nev: number of eigenvalues to compute
    :param gamma: kernel regularization parameter
    :return: les_desc: les descriptor [1 x nev] of data
    """

    h_op = _build_graph(data, sigma)
    les_desc, _ = _compute_log_eigenvalues(h_op, nev, gamma)
    return les_desc


def les_dist_comp(les_desc1, les_desc2):
    """
    Compute the LES distance

    :param les_desc1: LES descriptor of dataset1
    :param les_desc2: LES descriptor of dataset2
    :return: les_dist: les distance between the two datasets
    """

    return np.sqrt(np.sum((les_desc1 - les_desc2) ** 2))
   