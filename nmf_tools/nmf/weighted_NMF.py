import time
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _beta_loss_to_float, trace_dot, _check_init, norm
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
import numpy as np
import scipy.sparse as sp
import time
import warnings


from sklearn._config import config_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_is_fitted, check_non_negative
try:
    from sklearn.utils.validation import validate_data  # sklearn >= 1.6
except ImportError:
    validate_data = None


EPSILON = np.finfo(np.float32).eps


class WeightedNMF(NMF):

    def fit_transform(self, X, y=None, W=None, H=None, W_weights=None, H_weights=None):

        self._validate_params()

        validate_kwargs = dict(accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])
        if validate_data is not None:
            X = validate_data(self, X, **validate_kwargs)
        else:
            X = self._validate_data(X, **validate_kwargs)
        if W_weights is not None and self.beta_loss != 'frobenius':
            raise NotImplementedError
        with config_context(assume_finite=True):
            W, H, n_iter = self._fit_transform(X, W=W, H=H, W_weights=W_weights, H_weights=H_weights)

        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True, wX=None,
            W_weights=W_weights, H_weights=H_weights
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter

        return W
    
    def _check_w_h(self, X, W, H, update_H):
        """Check W and H, or initialize them."""
        n_samples, n_features = X.shape

        if self.init == "custom" and update_H:
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, self._n_components), "NMF (input W)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError(
                    "H and W should have the same dtype as X. Got "
                    "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
                )

        elif not update_H:

            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype:
                raise TypeError(
                    "H should have the same dtype as X. Got H.dtype = {}.".format(
                        H.dtype
                    )
                )
            
            # 'mu' solver should not be initialized by zeros
            if self.solver == "mu":
                if W is None:
                    avg = np.sqrt(X.mean() / self._n_components)
                    W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                raise NotImplementedError

        else:
            if W is not None or H is not None:
                warnings.warn(
                    (
                        "When init!='custom', provided W or H are ignored. Set "
                        " init='custom' to use them as initialization."
                    ),
                    RuntimeWarning,
                )

            if self._n_components == "auto":
                self._n_components = X.shape[1]

            W, H = _initialize_nmf(
                X, self._n_components, init=self.init, random_state=self.random_state
            )

        return W, H


    def reconstruction_error(self, X, W, H, W_weights=None, H_weights=None):
        """Calculate the NMF reconstruction error."""
        return _beta_divergence(
            X, W, H, self._beta_loss, square_root=True,
            W_weights=W_weights, H_weights=H_weights
        )


    def transform(self, X, W_weights=None, H_weights=None):
        check_is_fitted(self)
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
        )

        with config_context(assume_finite=True):
            W, *_ = self._fit_transform(X, H=self.components_, update_H=False, W_weights=W_weights, H_weights=H_weights)

        return W

    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True, W_weights=None, H_weights=None, error_at_init=None):
        check_non_negative(X, "NMF (input X)")
        
        # check parameters
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H)
        # scale the regularization terms
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)

        if self.solver == "cd":
            raise NotImplementedError
        elif self.solver == "mu":
            W, H, n_iter, *_ = _fit_multiplicative_update(
                X,
                W,
                H,
                self._beta_loss,
                self.max_iter,
                self.tol,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H,
                self.verbose,
                W_weights=W_weights,
                H_weights=H_weights,
                error_at_init=error_at_init
            )
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence."
                % self.max_iter,
                ConvergenceWarning,
            )

        return W, H, n_iter


def _fit_multiplicative_update(
    X,
    W,
    H,
    beta_loss="frobenius",
    max_iter=200,
    tol=1e-4,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
    error_at_init=None,
    W_weights=None,
    H_weights=None
):
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1.0 / (2.0 - beta_loss)
    elif beta_loss > 2:
        gamma = 1.0 / (beta_loss - 1.0)
    else:
        gamma = 1.0

    if W_weights is None:
        wX = X
    else:
        wX = get_wX(X, W_weights, H_weights)

    # used for the convergence criterion
    if error_at_init is None:
        error_at_init = _beta_divergence(
            X, W, H, beta_loss, 
            square_root=True,
            wX=wX,
            W_weights=W_weights, H_weights=H_weights
        )
        print('Error at init', error_at_init, flush=True)
    else:
        print("Assuming error at init=", error_at_init, flush=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None

    for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        W, H_sum, HHt, XHt = _multiplicative_update_w(
            wX,
            W,
            H,
            beta_loss=beta_loss,
            l1_reg_W=l1_reg_W,
            l2_reg_W=l2_reg_W,
            gamma=gamma,
            H_sum=H_sum,
            HHt=HHt,
            XHt=XHt,
            update_H=update_H,
            W_weights=W_weights,
            H_weights=H_weights
        )

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < EPSILON] = 0.0

        # update H (only at fit or fit_transform)
        if update_H:
            H = _multiplicative_update_h(
                wX,
                W,
                H,
                beta_loss=beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=gamma,
                W_weights=W_weights,
                H_weights=H_weights
            )

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < EPSILON] = 0.0

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(
                X, W, H, beta_loss,
                square_root=True,
                wX=wX,
                W_weights=W_weights,
                H_weights=H_weights
            )

            if verbose:
                iter_time = time.time()
                print(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_iter, iter_time - start_time, error),
                    flush=True
                )

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print(
            "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time),
            flush=True
        )

    return W, H, n_iter


def _multiplicative_update_w(
    X,
    W,
    H,
    beta_loss,
    l1_reg_W,
    l2_reg_W,
    gamma,
    H_sum=None,
    HHt=None,
    XHt=None,
    update_H=True,
    W_weights=None,
    H_weights=None
):
    """Update W in Multiplicative Update NMF."""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            XHt = safe_sparse_dot(X, H.T)
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XHt.copy()

        # Denominator
        if HHt is None:
            if H_weights is not None:
                HHt = np.dot(H_weights * H, H.T)
            else:
                HHt = np.dot(H, H.T)

        if W_weights is None:
            denominator = np.dot(W, HHt)
        else:
            denominator = np.dot(W_weights * W, HHt)
    else:
        raise NotImplementedError

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    W *= delta_W

    return W, H_sum, HHt, XHt


def _multiplicative_update_h(
    X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None,
    W_weights=None, H_weights=None
):
    """update H in Multiplicative Update NMF."""
    if beta_loss == 2:
        numerator = safe_sparse_dot(W.T, X)
        if W_weights is None:
            denominator = np.linalg.multi_dot([W.T, W, H])
        else:
            denominator = np.linalg.multi_dot([W.T, W_weights * W, H * H_weights])

    else:
        raise NotImplementedError

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON

    if A is not None and B is not None:
        # Updates for the online nmf
        if gamma != 1:
            H **= 1 / gamma
        numerator *= H
        A *= rho
        B *= rho
        A += numerator
        B += denominator
        H = A / B

        if gamma != 1:
            H **= gamma
    else:
        delta_H = numerator
        delta_H /= denominator
        if gamma != 1:
            delta_H **= gamma
        H *= delta_H

    return H


def get_wX(X, W_weights, H_weights):
    if sp.issparse(X):
        wX = X.multiply(H_weights.squeeze()).T.multiply(W_weights.squeeze()).T.tocsr()
    else:
        wX = X * W_weights * H_weights
    return wX


def _beta_divergence(X, W, H, beta, square_root=False, wX=None, W_weights=None, H_weights=None):
    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if sp.issparse(X):
            if W_weights is None:
                norm_X = np.dot(X.data, X.data)
                norm_WH = trace_dot(np.linalg.multi_dot([W.T, W, H]), H)
                cross_prod = trace_dot((X @ H.T), W)
            else:
                if wX is None:
                    wX = get_wX(X, W_weights, H_weights)
                sqrt_wW = W * np.sqrt(W_weights)
                sqrt_wH = H * np.sqrt(H_weights)

                norm_X = np.dot(wX.data, X.data)
                norm_WH = trace_dot(np.linalg.multi_dot([sqrt_wW.T, sqrt_wW, sqrt_wH]), sqrt_wH)
                cross_prod = trace_dot((wX @ H.T), W)

            res = (norm_X + norm_WH - 2.0 * cross_prod) / 2.0

        else:
            if W_weights is None:
                res = squared_norm(X - np.dot(W, H)) / 2.0
            else:
                res = squared_norm(
                    (X - np.dot(W, H)) * np.sqrt(W_weights) * np.sqrt(H_weights)
                ) / 2.0

        if square_root:
            if res < 0:
                print(res, norm_X, norm_WH, cross_prod)
            res = max(res, 0)
            return np.sqrt(res * 2)
        else:
            return res
    else:
        raise NotImplementedError


def _initialize_nmf(
        X, n_components, init=None, eps=1e-6, random_state=None, 
        W_weights=None, H_weights=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvda"
        else:
            init = "random"

    # Random initialization
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(
            X.dtype, copy=False
        )
        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(
            X.dtype, copy=False
        )
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H
