import numpy as np
import logging

logger = logging.getLogger(__name__)

class NewtonSolver:
    """
    A manual implementation of the Newton-Raphson method for solving systems of nonlinear equations.
    Supports bounded variables and damped updates.
    """
    def __init__(self, tol=1e-8, max_iter=100, damper=1.0, epsilon=1e-8):
        self.tol = tol
        self.max_iter = max_iter
        self.damper = damper
        self.epsilon = epsilon  # Step size for finite difference Jacobian

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
                return SolverResult(x, True, f"Converged in {k} iterations", k, k*n)
                
            # 2. Calculate Jacobian (Finite Difference)
            J = np.zeros((n, n))
            F_base = F
            
            # Identify active bounds (simple approach: just compute full J)
            # Optimization: If x_i is at bound and F_i pushes it further out, we could simplify.
            # But standard NR computes full J.
            
            for j in range(n):
                x_perturbed = x.copy()
                step = self.epsilon * max(abs(x[j]), 1.0) # Relative step size
                x_perturbed[j] += step
                
                F_perturbed = func(x_perturbed)
                J[:, j] = (F_perturbed - F_base) / step
            
            # 3. Solve Linear System: J * delta = -F
            # Use least squares if J is singular or ill-conditioned
            try:
                # delta = np.linalg.solve(J, -F) # Faster but can fail
                delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=None) # More robust
            except np.linalg.LinAlgError:
                return SolverResult(x, False, f"Singular Jacobian at iter {k}", k, k*n)
            
            # 4. Update with Damping
            x_new = x + self.damper * delta
            
            # 5. Apply Bounds
            if bounds:
                lower, upper = bounds
                x_new = np.clip(x_new, lower, upper)
            
            # Check step size for stagnation
            step_norm = np.linalg.norm(x_new - x)
            if step_norm < 1e-14:
                 return SolverResult(x_new, True, f"Stagnation (step too small) at iter {k}", k, k*n)

            x = x_new
            
            # Optional: Log progress
            # logger.debug(f"Iter {k}: Max Res={max_res:.2e}, Step={step_norm:.2e}")

        return SolverResult(x, False, "Max iterations reached", self.max_iter, self.max_iter*n)

class SolverResult:
    def __init__(self, x, success, message, nit, nfev):
        self.x = x
        self.success = success
        self.message = message
        self.nit = nit      # Number of iterations
        self.nfev = nfev    # Number of function evaluations (approx)
        self.cost = np.linalg.norm(x) # Placeholder, usu. norm(F)
