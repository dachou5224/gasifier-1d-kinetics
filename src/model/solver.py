import numpy as np
import logging

from .jax_residual_adapter import finite_difference_jacobian_centered

logger = logging.getLogger(__name__)

class NewtonSolver:
    """
    A manual implementation of the Newton-Raphson method for solving systems of nonlinear equations.
    Supports bounded variables and damped updates.
    """
    def __init__(
        self,
        tol=1e-8,
        max_iter=100,
        damper=1.0,
        epsilon=1e-8,
        jacobian="finite_difference",
    ):
        """
        jacobian:
            - 'finite_difference': 前向有限差分（与原实现一致）
            - 'centered': 中心差分（与 jax_residual_adapter 一致，更稳）
        """
        self.tol = tol
        self.max_iter = max_iter
        self.damper = damper
        self.epsilon = epsilon  # Step size for finite difference Jacobian
        if jacobian not in ("finite_difference", "centered"):
            raise ValueError("jacobian must be 'finite_difference' or 'centered'")
        self.jacobian = jacobian

    def solve(self, func, x0, bounds=None):
        """
        Solves func(x) = 0 starting from x0.
        
        Args:
            func: Callable that takes x and returns residuals array.
            x0: Initial guess array.
            bounds: Tuple (lower, upper) of arrays.
            
        Returns:
            Solution object with .x, .success, .message, .nfev, .njev attributes.
        """
        x = np.array(x0, dtype=float)
        n = len(x)
        
        # Initial bounds check
        if bounds:
            lower, upper = bounds
            x = np.clip(x, lower, upper)
        
        for k in range(self.max_iter):
            # 1. Calculate Residuals
            F = func(x)
            res_norm = np.linalg.norm(F)
            max_res = np.max(np.abs(F))
            
            # Check convergence
            if max_res < self.tol:
                cost = 0.5 * np.sum(F**2)
                return SolverResult(x, True, f"Converged in {k} iterations", k, k*n, cost=cost)
                
            # 2. Jacobian (前向或中心差分)
            if self.jacobian == "centered":
                J = finite_difference_jacobian_centered(func, x, n_vars=n, epsilon=self.epsilon)
            else:
                J = np.zeros((n, n))
                F_base = F
                for j in range(n):
                    x_perturbed = x.copy()
                    step = self.epsilon * max(abs(x[j]), 1.0)
                    x_perturbed[j] += step
                    F_perturbed = func(x_perturbed)
                    J[:, j] = (F_perturbed - F_base) / step
            
            # 3. Solve Linear System: J * delta = -F
            # Use least squares if J is singular or ill-conditioned
            try:
                # delta = np.linalg.solve(J, -F) # Faster but can fail
                delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=None) # More robust
            except np.linalg.LinAlgError:
                cost = 0.5 * np.sum(F**2)
                return SolverResult(x, False, f"Singular Jacobian at iter {k}", k, k*n, cost=cost)
            
            # 4. Update with Damping
            x_new = x + self.damper * delta
            
            # 5. Apply Bounds
            if bounds:
                lower, upper = bounds
                x_new = np.clip(x_new, lower, upper)
            
            # Check step size for stagnation
            step_norm = np.linalg.norm(x_new - x)
            if step_norm < 1e-14:
                 F_new = func(x_new)
                 max_res_new = np.max(np.abs(F_new))
                 cost = 0.5 * np.sum(F_new**2)
                 if max_res_new < self.tol:
                     return SolverResult(x_new, True, f"Stagnation (converged) at iter {k}", k, k*n, cost=cost)
                 return SolverResult(
                     x_new,
                     False,
                     f"Stagnation (step too small) at iter {k} with residual max|F|={max_res_new:.3e}",
                     k,
                     k*n,
                     cost=cost,
                 )

            x = x_new
            
            # Optional: Log progress
            # logger.debug(f"Iter {k}: Max Res={max_res:.2e}, Step={step_norm:.2e}")

        F_final = func(x)
        cost = 0.5 * np.sum(F_final**2)
        return SolverResult(x, False, "Max iterations reached", self.max_iter, self.max_iter*n, cost=cost)

class SolverResult:
    def __init__(self, x, success, message, nit, nfev, cost=None, njev=None):
        self.x = x
        self.success = success
        self.message = message
        self.nit = nit      # Number of iterations
        self.nfev = nfev    # Number of function evaluations (approx)
        self.njev = njev    # Jacobian evaluations (optional)
        self.cost = cost if cost is not None else 0.5 * np.sum(x**2) # Should not fallback to norm(x) roughly, but safe 
