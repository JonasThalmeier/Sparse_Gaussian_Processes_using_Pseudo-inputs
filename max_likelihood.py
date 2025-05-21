import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import value_and_grad
from scipy.optimize import minimize
from jax.scipy.stats import multivariate_normal
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
import numpy as np

class SparseGPModel:
    """
    Sparse Gaussian Process model using inducing points for scalable inference.
    
    Attributes:
        N (int): Number of training points
        D (int): Dimension of input space
        M (int): Number of inducing points
        margin (float): Margin for inducing points
    """
    def __init__(self, N, D, M, margin=0.5):
        self.N = N  # Number of data points
        self.D = D  # Input dimension
        self.M = M  # Number of inducing points
        self.margin = margin  # Margin for inducing points
        
    @partial(jax.jit, static_argnums=0)
    def kernel(self, X1, X2, log_c, log_b):
        """
        Compute RBF kernel between two sets of points.
        
        Args:
            X1: First set of points (N x D)
            X2: Second set of points (M x D)
            log_c: Log of kernel amplitude
            log_b: Log of length scales
        
        Returns:
            Kernel matrix of shape (N x M)
        """
        c = jnp.exp(log_c)  # Kernel amplitude
        b = jnp.exp(log_b)  # Length scales
        diff = X1[:, None, :] - X2[None, :, :]  # Pairwise differences
        sq_dist = jnp.sum(b * (diff ** 2), axis=2)  # Weighted distances
        return c * jnp.exp(-0.5 * sq_dist)  # RBF kernel
    
    def pack_params(self, X_bar, log_c, log_b, log_sigma_sq):
        """
        Pack all parameters into a single vector for optimization.
        
        Args:
            X_bar: Inducing points (M x D)
            log_c: Log kernel amplitude
            log_b: Log length scales
            log_sigma_sq: Log noise variance
        
        Returns:
            Concatenated parameter vector
        """
        return jnp.concatenate([X_bar.ravel(), log_c[None], log_b, log_sigma_sq[None]])
    
    @partial(jax.jit, static_argnums=0)
    def unpack_params(self, params):
        """
        Unpack parameters from optimization vector into separate components.
        
        Args:
            params: Concatenated parameter vector
        
        Returns:
            Tuple of (X_bar, log_c, log_b, log_sigma_sq)
        """
        X_bar = params[:self.M*self.D].reshape(self.M, self.D)
        log_c = params[self.M*self.D]
        log_b = params[self.M*self.D + 1:self.M*self.D + 1 + self.D]
        log_sigma_sq = params[self.M*self.D + 1 + self.D]
        return X_bar, log_c, log_b, log_sigma_sq
    
    @partial(jax.jit, static_argnums=0)
    def neg_log_likelihood(self, params, X, y, jitter=1e-6):
        """
        Computes the negative log marginal likelihood of the sparse GP model.
        
        This implementation follows Snelson & Ghahramani (2006) using the sparse
        approximation with M inducing points. The computation uses Cholesky
        decomposition for numerical stability and includes an optional
        regularization term to encourage spread-out inducing points.
        
        Args:
            params: Packed parameter vector containing:
                - X_bar: Inducing point locations (M x D)
                - log_c: Log of kernel amplitude
                - log_b: Log of length scales
                - log_sigma_sq: Log of noise variance
            X: Input training points (N x D)
            y: Target values (N,)
            jitter: Small value added to diagonal for numerical stability
        
        Returns:
            Negative log marginal likelihood value (scalar)
        """
        X_bar, log_c, log_b, log_sigma_sq = self.unpack_params(params)
        c = jnp.clip(jnp.exp(log_c), 1e-6, 1e6)  # Clip to avoid extreme values
        sigma_sq = jnp.clip(jnp.exp(log_sigma_sq), 1e-6, 1e6)
        
        # Add more jitter to improve conditioning
        jitter = 1e-4
        
        # Compute kernels with careful conditioning
        K_MM = self.kernel(X_bar, X_bar, log_c, log_b)
        K_MM = K_MM + jitter * jnp.eye(self.M)
        K_NM = self.kernel(X, X_bar, log_c, log_b)
        
        #X_bar = X_bar + jitter * jnp.eye(self.M)  # Add jitter to X_bar for numerical stability

        # Use Cholesky decomposition instead of direct inverse
        L_MM = jnp.linalg.cholesky(K_MM)  # (M×M)

        # V = L_MM^{-1} K_NM^T  => shape (M, N)
        V = jax.scipy.linalg.solve_triangular(
            L_MM, K_NM.T, lower=True
        )
        
        # ---- Compute Lambda diagonal ----
        # K_NN diagonal (for e.g. RBF it's simply c + noise maybe)
        K_NN_diag = c #kernel_diag(X, log_c, log_b)  # returns shape (N,)
        lambda_diag = K_NN_diag - jnp.sum(V**2, axis=0)  # shape (N,)

        # Total diagonal noise term D = λ + σ²
        D = lambda_diag + sigma_sq

        # ---- Build full N×N covariance C ----
        # Q_NN = (K_NM K_MM^{-1} K_MN) = V^T V
        Q_NN = (V.T @ V)  # shape (N, N)
        C = Q_NN + jnp.diag(D)  # shape (N, N)

        # ---- Cholesky decomposition of C ----
        L_C = jnp.linalg.cholesky(C)  # shape (N, N)

        # ---- Log determinant and quadratic form ----
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_C)))
        alpha = jax.scipy.linalg.solve_triangular(
            L_C.T,
            jax.scipy.linalg.solve_triangular(L_C, y, lower=True),
            lower=False
        )
        quad = jnp.dot(y, alpha)

        # ---- Negative log likelihood ----
        nll = 0.5 * (logdet + quad + self.N * jnp.log(2 * jnp.pi))


        #-----------Regularization-------------
        diffs = X_bar[:, None, :] - X_bar[None, :, :]
        sq_dists = jnp.sum(diffs ** 2, axis=-1)
        # Add small value to denominator to avoid division by zero
        reg = jnp.sum(1.0 / (sq_dists + 1e-6)) - X_bar.shape[0]  # exclude self-distances
        reg_weight =0#1e-6  # You can tune this weight
        nll += reg_weight * reg

        return nll
    
    def f_bar(self, jitter=1e-6):
        """
        Computes the posterior mean and covariance at the inducing points.
        
        This method implements the key equations from the SPGP model to compute
        the posterior distribution over the latent function values at the
        inducing points (pseudo-targets). It uses the optimized hyperparameters
        and inducing point locations from training.
        
        The computation includes:
        1. Kernel matrix computations (K_M, K_NM)
        2. Lambda (diagonal correction) computation
        3. Q matrix (M x M) for posterior
        4. Posterior mean and covariance at inducing points
        
        Args:
            jitter: Small value added to diagonal for numerical stability
        
        Returns:
            Tuple of:
            - mean: Posterior mean at inducing points (M,)
            - cov: Posterior covariance at inducing points (M x M)
        """
        # Cast shapes explicitly for clarity
        c = jnp.exp(self.log_c_opt)  # scalar
        sigma_sq = jnp.exp(self.log_sigma_sq_opt)  # scalar
        
        # Compute kernel matrices with explicit shapes
        self.K_M = self.kernel(self.X_bar_opt, self.X_bar_opt, self.log_c_opt, self.log_b_opt)  # (M, M)
        self.K_M = self.K_M + jitter * jnp.eye(self.M)  # (M, M)
        self.K_NM = self.kernel(self.X, self.X_bar_opt, self.log_c_opt, self.log_b_opt)  # (N, M)
        
        # Compute inverse of K_M
        self.K_M_inv = jnp.linalg.inv(self.K_M)  # (M, M)
        
        # Compute quadratic terms efficiently
        quad_terms = jnp.sum(self.K_NM @ self.K_M_inv * self.K_NM, axis=1)  # (N,)
        
        # Compute lambda diagonal
        lambda_diag = c - quad_terms  # (N,)
        self.inv_lambda_plus_sigma = 1.0 / (lambda_diag + sigma_sq)  # (N,)
        
        # Compute Q matrix
        Q = self.K_M + self.K_NM.T @ jnp.diag(self.inv_lambda_plus_sigma) @ self.K_NM  # (M, M)
        self.Q_inv = jnp.linalg.inv(Q)  # (M, M)
        
        # Compute mean and covariance
        noise_matrix = jnp.diag(lambda_diag) + sigma_sq * jnp.eye(self.N)  # (N, N)
        mean = self.K_M @ self.Q_inv @ self.K_NM.T @ jnp.linalg.solve(noise_matrix, self.y)  # (M,)
        cov = self.K_M @ self.Q_inv @ self.K_M  # (M, M)
        
        return mean, cov

    def fit(self, X, y):
        """
        Fits the Sparse GP model by optimizing inducing points and hyperparameters.
        
        This method implements the training procedure for the SPGP model:
        1. Initializes inducing points and hyperparameters
        2. Sets up parameter bounds for optimization
        3. Minimizes negative log likelihood using SLSQP optimizer
        4. Stores optimized parameters for prediction
        
        The optimization includes:
        - Inducing point locations (X_bar)
        - Kernel amplitude (log_c)
        - Length scales (log_b)
        - Noise variance (log_sigma_sq)
        
        Args:
            X: Training inputs (N x D)
            y: Training targets (N,)
        
        Returns:
            Tuple containing:
            - mean_f_bar: Posterior mean at inducing points (M,)
            - cov_f_bar: Posterior covariance at inducing points (M x M)
            - X_bar_opt: Optimized inducing points (M x D)
            - log_c_opt: Optimized log kernel amplitude
            - log_b_opt: Optimized log length scales (D,)
            - log_sigma_sq_opt: Optimized log noise variance
            - X_bar_init: Initial inducing points (M x D)
        """
        self.X = X
        self.y = y
        # Enable float64 support in JAX
        jax.config.update("jax_enable_x64", True)
        
        # Initialize parameters
        # X_bar_init = jax.random.uniform(jax.random.PRNGKey(0), shape=(M, D), minval=0.0, maxval=0.5)
        X_bar_init = jax.random.uniform(jax.random.PRNGKey(0), shape=(self.M, self.D), minval=X.min(), maxval=X.max())
        log_c_init = jnp.array(0.0)
        log_b_init = jnp.zeros(self.D)
        log_sigma_sq_init = jnp.array(-2.3)
        params_init = self.pack_params(X_bar_init, log_c_init, log_b_init, log_sigma_sq_init)


        X_min = X.min() - self.margin
        X_max = X.max() + self.margin
        # Set bounds for all parameters
        bounds = []
        # Bounds for X_bar (all inducing points)
        for _ in range(self.M*self.D):
            bounds.append((X_min, X_max))  # Allow X_bar to move in reasonable range
        # Bounds for log_c
        bounds.append((-5.0, 5.0))
        # Bounds for log_b
        for _ in range(self.D):
            bounds.append((-5.0, 5.0))
        # Bounds for log_sigma_sq
        bounds.append((-10.0, 0.0))

        # Create value and gradient function with debug print
        def neg_log_likelihood_fn(params):
            nll = self.neg_log_likelihood(params, X, y)
            if jnp.isnan(nll):
                print("Warning: NaN in negative log likelihood")
            return nll

        obj_grad = jax.value_and_grad(neg_log_likelihood_fn)

        # Define objective for scipy optimizer
        def objective(params_np):
            params_jax = jnp.array(params_np)
            value, grad = obj_grad(params_jax)
            if jnp.any(jnp.isnan(grad)):
                print("Warning: NaN in gradient")
            return np.array(value), np.array(grad)

        # Run optimization with bounds and more iterations
        result = minimize(
            lambda p: objective(p)[0],
            params_init,
            # method='L-BFGS-B',
            method='SLSQP',
            jac=lambda p: objective(p)[1],
            bounds=bounds,
            options={
                'maxiter': 100000,  # Increase maximum iterations
                'maxfun': 20000,    # Increase maximum function evaluations
                'ftol': 1e-8,      # Add convergence tolerance
                'gtol': 1e-7       # Add gradient tolerance
            }
        )

        print(f"Optimization status: {result.message}")
        print(f"Final loss: {result.fun}")
        self.X_bar_opt, self.log_c_opt, self.log_b_opt, self.log_sigma_sq_opt = self.unpack_params(jnp.array(result.x))
        self.mean_f_bar, self.cov_f_bar = self.f_bar()
        return self.mean_f_bar, self.cov_f_bar, self.X_bar_opt, self.log_c_opt, self.log_b_opt, self.log_sigma_sq_opt, X_bar_init


    def predict(self, X_test):
        """
        Predicts mean and variance for test points using the trained SPGP model.
        
        This method implements the sparse GP prediction equations from 
        Snelson & Ghahramani (2006). It computes both the predictive mean
        and variance efficiently using the pre-computed matrices from training.
        
        The prediction uses the deterministic inducing point approximation:
        mean = k*ᵀ Q⁻¹ Kₘₙᵀ Λ⁻¹ y
        var = k** - k*ᵀ (Kₘ⁻¹ - Q⁻¹) k* + σ²
        
        Args:
            X_test: Test input points (N_test x D)
        
        Returns:
            Tuple of:
            - mean: Predictive mean at test points (N_test,)
            - var: Predictive variance at test points (N_test,)
            
        Note:
            Requires the model to be fitted first (self.X_bar_opt etc. must exist)
        """
        sigma_sq = jnp.exp(self.log_sigma_sq_opt)
        
        # Compute required kernel matrices
        k_star = self.kernel(X_test, self.X_bar_opt, self.log_c_opt, self.log_b_opt)
        K_starstar = self.kernel(X_test, X_test, self.log_c_opt, self.log_b_opt)
        
        # Compute predictive mean
        mean = k_star @ self.Q_inv @ self.K_NM.T @ jnp.diag(self.inv_lambda_plus_sigma) @ self.y
        
        # Compute predictive variance
        var = jnp.diag(K_starstar - k_star @ (self.K_M_inv - self.Q_inv) @ k_star.T + sigma_sq)
        
        return mean, var
