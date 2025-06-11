import torch
import torch.nn as nn
from torchdiffeq._impl.solvers import AdaptiveStepsizeODESolver
from torchdiffeq._impl.misc import _select_initial_step, _handle_unused_kwargs, _rms_norm
from torchdiffeq._impl.interp import _interp_evaluate, _interp_fit
import math
import time
from typing import Optional, Callable, Tuple, List
import collections

# --- Custom GMRES Implementation ---
# This section provides a pure-PyTorch implementation of the GMRES algorithm,
# removing the dependency on `torch.linalg.gmres`.

def _apply_preconditioner(preconditioner, x):
    """Applies the preconditioner to a vector."""
    if preconditioner is None:
        return x
    return preconditioner(x)

def _arnoldi_process(A, v1, k):
    """
    Performs k steps of the Arnoldi iteration to build an orthonormal basis
    for the Krylov subspace. This is the core of the GMRES method.

    Args:
        A (callable): The linear operator.
        v1 (torch.Tensor): The starting vector.
        k (int): The number of Arnoldi iterations.

    Returns:
        A tuple (V, H, j), where V is the orthonormal basis, H is the
        Hessenberg matrix, and j is the actual number of iterations performed
        (can be less than k if the subspace is exhausted).
    """
    V = torch.zeros(v1.shape[0], k + 1, device=v1.device, dtype=v1.dtype)
    H = torch.zeros(k + 1, k, device=v1.device, dtype=v1.dtype)
    V[:, 0] = v1

    for j in range(k):
        w = A(V[:, j])
        for i in range(j + 1):
            H[i, j] = torch.dot(w, V[:, i])
            w = w - H[i, j] * V[:, i]
        
        h_next = torch.linalg.norm(w)
        H[j + 1, j] = h_next
        
        # If h_next is close to zero, the Krylov subspace is invariant,
        # and we can stop the iteration early.
        if h_next < 1e-12:
            return V[:, :j + 2], H[:j + 2, :j + 1], j + 1
        
        V[:, j + 1] = w / h_next

    return V, H, k

def gmres(A, b, x0=None, M=None, restart=20, maxiter=None, atol=1e-5):
    """
    Custom implementation of the Generalized Minimal RESidual method (GMRES).

    Solves the linear system Ax = b.

    Args:
        A (callable): A function that computes the matrix-vector product A(v).
        b (torch.Tensor): The right-hand side vector.
        x0 (torch.Tensor, optional): The initial guess. Defaults to zeros.
        M (callable, optional): Preconditioner. A function that computes M_inv(v).
        restart (int, optional): Number of iterations before restarting.
        maxiter (int, optional): Maximum number of outer iterations.
        atol (float, optional): Absolute tolerance for convergence.

    Returns:
        A tuple (x, iters), where x is the solution and iters is the total
        number of iterations. `iters` is negative if the method did not converge.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone().detach()

    if maxiter is None:
        maxiter = 2 * b.shape[0]

    # Preconditioned operator for Arnoldi iteration
    def preconditioned_A(v):
        return _apply_preconditioner(M, A(v))

    total_iters = 0
    # Outer loop for restarted GMRES
    for outer_iter in range(maxiter):
        # Calculate initial residual
        r = _apply_preconditioner(M, b - A(x))
        r_norm = torch.linalg.norm(r)

        # Check for convergence
        if r_norm < atol:
            return x, total_iters

        # Normalize residual to get the first Krylov vector
        v1 = r / r_norm
        
        # Perform Arnoldi iteration
        V, H, actual_k = _arnoldi_process(preconditioned_A, v1, restart)
        total_iters += actual_k
        
        # Form the right-hand side for the least-squares problem
        e1 = torch.zeros(actual_k + 1, device=b.device, dtype=b.dtype)
        e1[0] = r_norm

        # Solve the small least-squares problem H*y = e1
        y, _, _, _ = torch.linalg.lstsq(H, e1)
        
        # Update the solution
        x = x + V[:, :actual_k] @ y
        
        # Check for convergence again after update (important for restarts)
        r_final = b - A(x)
        if torch.linalg.norm(r_final) < atol:
            return x, total_iters

    # If we reached here, the method did not converge
    return x, -total_iters

# --- State Management ---
# Using a namedtuple for state management is a clean pattern adopted from
# the torchdiffeq library itself. It makes the code more readable.
_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')

# --- Main Solver Class ---

class EnhancedNeuralODESolver(AdaptiveStepsizeODESolver):
    """
    Fully corrected and enhanced 3rd-order L-stable Rosenbrock solver for
    neural ODE training on stiff inverter systems.
    """
    
    order = 3
    
    # Rodas3 coefficients
    gamma = 0.5
    a21 = 2.0
    a31 = 48.0 / 25.0
    a32 = 6.0 / 25.0
    c2 = 1.0
    c3 = 1.0
    g21 = -1.0
    g31 = -24.0 / 25.0
    g32 = -3.0 / 25.0
    b1 = 19.0 / 9.0
    b2 = 0.5
    b3 = 2.0 / 9.0
    e1 = 1.0
    e2 = -1.0
    e3 = 0.0
    
    def __init__(self, func, y0, rtol=1e-7, atol=1e-9, **kwargs):
        # List of arguments expected by AdaptiveStepsizeODESolver
        parent_args = ['dtype', 'norm']
        parent_kwargs = {k: kwargs[k] for k in parent_args if k in kwargs}
        if 'dtype' not in parent_kwargs:
            parent_kwargs['dtype'] = y0.dtype
        # Call parent constructor with only expected arguments
        super().__init__(func=func, y0=y0, rtol=rtol, atol=atol, **parent_kwargs)

        # Store all options for this solver (including custom ones)
        self.options = kwargs
        self.func = func
        self.rtol = rtol
        self.atol = atol
        self.device = y0.device
        self.dtype = y0.dtype
        
        # Initialize mixed precision settings
        self.use_mixed_precision = kwargs.get('use_mixed_precision', False)
        if self.use_mixed_precision:
            self.compute_dtype = torch.float32
            self.storage_dtype = y0.dtype
        else:
            self.compute_dtype = y0.dtype
            self.storage_dtype = y0.dtype
        
        # Initialize preconditioner settings
        self.use_preconditioning = kwargs.get('use_preconditioning', True)
        self.precond_type = kwargs.get('precond_type', 'block_diagonal')
        self.precond_block_sizes = kwargs.get('precond_block_sizes', None)
        self.precond_refresh_rate = kwargs.get('precond_refresh_rate', 10)
        self.precond_regularization = kwargs.get('precond_regularization', 1e-6)
        
        # Initialize GMRES settings
        self.gmres_restart = kwargs.get('gmres_restart', 20)
        self.gmres_max_iter = kwargs.get('gmres_max_iter', 100)
        self.gmres_tol = kwargs.get('gmres_tol', 1e-5)
        self.gmres_adaptive = kwargs.get('gmres_adaptive', True)
        self.gmres_tol_factor = kwargs.get('gmres_tol_factor', 0.1)
        self.gmres_tol_current = self.gmres_tol
        self.last_error_norm = 1.0
        
        # Initialize gradient monitoring settings
        self.monitor_gradient_norm = kwargs.get('monitor_gradient_norm', False)
        self.gradient_norm_history = []
        self.condition_number_history = []
        self.condition_number_threshold = kwargs.get('condition_number_threshold', 1e6)
        self.adaptive_regularization = kwargs.get('adaptive_regularization', False)
        
        # Initialize other settings
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self.gradient_warmup_steps = kwargs.get('gradient_warmup_steps', 0)
        self.jacobian_autodiff_threshold = kwargs.get('jacobian_autodiff_threshold', 5)
        
        # Initialize state
        self.preconditioner = None
        self.num_accepted_steps = 0
        self.consecutive_rejects = 0
        self.total_jacobian_evals = 0
        self.total_steps = 0
        
        # --- Unpack all solver options with sensible defaults ---
        self.min_step_size = kwargs.get('min_step_size', 1e-12)
        self.max_step_size = kwargs.get('max_step_size', float('inf'))
        self.safety_factor = kwargs.get('safety_factor', 0.9)
        self.min_factor = kwargs.get('min_factor', 0.2)
        self.max_factor = kwargs.get('max_factor', 10.0)
        self.max_num_steps = kwargs.get('max_num_steps', 2**31 - 1)
        
        # Event detection settings
        self.event_fn = kwargs.get('event_fn')
        self.event_tol = kwargs.get('event_tol', 1e-4)
        
        # Gradient preservation settings
        self.gradient_norm_clip = kwargs.get('gradient_norm_clip', 100.0)
        self.gradient_warmup_steps = kwargs.get('gradient_warmup_steps', 100)
        
        # Internal state
        self.rk_state = None
    
    def _before_integrate(self, t):
        """Initializes the solver state before integration."""
        f0 = self.func(t[0], self.y0)
        initial_dt = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol, self.norm, f0=f0)
        
        # Initialize rk_state with dummy interpolation coefficients
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], initial_dt, [self.y0] * 5)
        
        # Reset diagnostics and other internal states
        self.preconditioner = None
        self.last_event_values = None
        self.last_error_norm = 1.0
        self.gmres_tol_current = self.gmres_tol
        self.consecutive_rejects = 0
        self.total_steps = 0
        self.total_gmres_iters = 0
        self.total_jacobian_evals = 0
        self.num_accepted_steps = 0
        self.num_rejected_steps = 0
        self.gradient_norm_history = []
        self.condition_number_history = []

    def _advance(self, next_t):
        """
        This is the critical missing method. It advances the solution to the
        next time point, `next_t`, by repeatedly calling `_adaptive_step`.
        """
        n_steps = 0
        while next_t > self.rk_state.t1:
            if n_steps > self.max_num_steps:
                raise RuntimeError(f'max_num_steps exceeded ({self.max_num_steps})')
            
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
            
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _interp_fit(self, y0, y1, f0, f1, dt):
        """Fit a cubic Hermite polynomial for dense output."""
        # This is a standard method for creating a C1-continuous interpolant.
        # It uses the function values and derivatives at the start and end of the step.
        dt = dt.type_as(y0)
        # y_mid is approximated for the 4th order polynomial fit
        y_mid = y0 + 0.5 * dt * f0 - 0.125 * dt * (f1 - f0)
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)

    def _adaptive_step(self, rk_state):
        """Performs a single adaptive step with error control."""
        y0, f0, _, t0, dt, _ = rk_state
        dt = torch.clamp(dt, self.min_step_size, self.max_step_size)
        t1 = t0 + dt

        y1, error, converged = self._step_func(self.func, t0, dt, y0, f0)

        if not converged:
            self.consecutive_rejects += 1
            self.num_rejected_steps += 1
            dt_next = dt * 0.25 # Aggressive reduction on GMRES failure
            return _RungeKuttaState(y0, f0, t0, t0, dt_next, rk_state.interp_coeff)

        # Error control
        error_tol = self.atol + self.rtol * torch.maximum(torch.abs(y0), torch.abs(y1))
        error_norm = self.norm(error / error_tol)

        if not torch.isfinite(error_norm):
            self.consecutive_rejects += 1
            self.num_rejected_steps += 1
            dt_next = dt * 0.1
            return _RungeKuttaState(y0, f0, t0, t0, dt_next, rk_state.interp_coeff)

        accept_step = error_norm <= 1.0
        
        # PI step size controller
        if error_norm < 1e-10:
            factor = self.max_factor
        else:
            order_p1 = self.order + 1
            factor = self.safety_factor * (1.0 / error_norm)**(0.7 / order_p1) * (self.last_error_norm / error_norm)**(0.4 / order_p1)

        factor = torch.clamp(torch.as_tensor(factor), self.min_factor, self.max_factor).item()

        if accept_step:
            t_next = t1
            y_next = y1
            self.num_accepted_steps += 1
            self.consecutive_rejects = 0
            self.last_error_norm = error_norm
            self.total_steps += 1
            # Evaluate f at the end of the step for interpolation
            f_next = self.func(t_next, y_next)
            interp_coeff = self._interp_fit(y0, y_next, f0, f_next, dt)
        else:
            t_next = t0
            y_next = y0
            f_next = f0
            factor = min(factor, 0.5)
            self.num_rejected_steps += 1
            self.consecutive_rejects += 1
            interp_coeff = rk_state.interp_coeff
        
        dt_next = dt * factor
        return _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)

    def _step_func(self, func, t0, dt, y0, f0):
        """Executes a single Rosenbrock step."""
        y0_compute = y0.to(self.compute_dtype)
        f0_compute = f0.to(self.compute_dtype)

        def linear_operator(v):
            jvp = self._compute_jvp(func, t0, y0_compute, v)
            return v - dt * self.gamma * jvp

        self._create_preconditioner(func, t0, y0_compute, dt)
        M_solve = self.preconditioner if self.preconditioner is not None else None
        
        base_tol = self.atol + self.rtol * self.norm(y0_compute)
        gmres_tol = self._adaptive_gmres_tolerance(base_tol)

        # Stage 1
        k1, iters1 = gmres(linear_operator, f0_compute, M=M_solve, atol=gmres_tol, restart=self.gmres_restart, maxiter=self.gmres_max_iter)
        if iters1 < 0: return None, None, False
        self.total_gmres_iters += abs(iters1)
        
        # Stage 2
        y2 = y0_compute + self.a21 * dt * k1
        f2 = func(t0 + self.c2 * dt, y2)
        rhs2 = f2 + self.g21 * k1 / (dt * self.gamma)
        k2, iters2 = gmres(linear_operator, rhs2, x0=k1, M=M_solve, atol=gmres_tol, restart=self.gmres_restart, maxiter=self.gmres_max_iter)
        if iters2 < 0: return None, None, False
        self.total_gmres_iters += abs(iters2)
        
        # Stage 3
        y3 = y0_compute + dt * (self.a31 * k1 + self.a32 * k2)
        f3 = func(t0 + self.c3 * dt, y3)
        rhs3 = f3 + (self.g31 * k1 + self.g32 * k2) / (dt * self.gamma)
        k3, iters3 = gmres(linear_operator, rhs3, x0=k2, M=M_solve, atol=gmres_tol, restart=self.gmres_restart, maxiter=self.gmres_max_iter)
        if iters3 < 0: return None, None, False
        self.total_gmres_iters += abs(iters3)

        y1 = y0_compute + dt * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3)
        error = dt * (self.e1 * k1 + self.e2 * k2 + self.e3 * k3)
        
        return y1.to(self.storage_dtype), error.to(self.storage_dtype), True

    # --- All other helper methods (`_compute_jvp`, `_estimate_jacobian_blocks`, etc.)
    # from the user's code are kept here as they contain the advanced logic.
    # They are omitted for brevity in this view but are part of the full implementation.
    def _compute_jvp(self, func, t, y, v):
        create_graph = self.training if hasattr(self, 'training') else y.requires_grad
        if self.total_steps < self.gradient_warmup_steps:
            clip_value = self.gradient_norm_clip * (self.total_steps / self.gradient_warmup_steps)
        else:
            clip_value = self.gradient_norm_clip
        with torch.enable_grad():
            y_dual = y.detach().requires_grad_(True)
            if self.gradient_checkpointing and create_graph:
                f_dual = torch.utils.checkpoint.checkpoint(func, t, y_dual, use_reentrant=False)
            else:
                f_dual = func(t, y_dual)
            if self.monitor_gradient_norm and create_graph and self.total_steps % 10 == 0:
                with torch.no_grad():
                    try:
                        grad = torch.autograd.grad(f_dual.sum(), y_dual, retain_graph=True, create_graph=False)[0]
                        grad_norm = grad.norm().item()
                        self.gradient_norm_history.append(grad_norm)
                        if self.adaptive_regularization and grad_norm > 1e6:
                            self.regularization_factor = min(1e-6, self.regularization_factor * 2)
                    except RuntimeError: pass
            jvp = torch.autograd.grad(f_dual, y_dual, v, create_graph=create_graph, retain_graph=create_graph)[0]
            if create_graph and clip_value > 0:
                jvp_norm = jvp.norm()
                if jvp_norm > clip_value:
                    jvp = jvp * (clip_value / (jvp_norm + 1e-8))
        return jvp

    def _estimate_jacobian_blocks(self, func, t, y):
        J_blocks = []
        y_flat = y.flatten()
        start_idx = 0
        for block_size in self.precond_block_sizes:
            end_idx = start_idx + block_size
            if block_size <= self.jacobian_autodiff_threshold:
                block_y = y_flat[start_idx:end_idx].clone().requires_grad_(True)
                def block_func(sub_y):
                    full_y = y.clone()
                    full_y.flatten()[start_idx:end_idx] = sub_y
                    return func(t, full_y).flatten()[start_idx:end_idx]
                with torch.no_grad():
                    J_block = torch.autograd.functional.jacobian(block_func, block_y, create_graph=False)
            else:
                J_block = self._finite_diff_jacobian_block(func, t, y, start_idx, end_idx)
            if self.use_mixed_precision:
                J_block = J_block.to(self.compute_dtype)
            J_blocks.append(J_block)
            start_idx = end_idx
        return J_blocks

    def _finite_diff_jacobian_block(self, func, t, y, start_idx, end_idx):
        block_size = end_idx - start_idx
        eps = (1e-7 * (1.0 + torch.abs(y).max())).item()
        J_block = torch.zeros(block_size, block_size, device=y.device, dtype=y.dtype)
        with torch.no_grad():
            f0 = func(t, y).flatten()[start_idx:end_idx]
            for i in range(block_size):
                y_pert = y.clone()
                y_pert.flatten()[start_idx + i] += eps
                f_pert = func(t, y_pert).flatten()[start_idx:end_idx]
                J_block[:, i] = (f_pert - f0) / eps
        return J_block

    def _create_preconditioner(self, func, t, y, dt):
        """Create preconditioner for the linear system."""
        # Get Jacobian blocks
        J_blocks = self._estimate_jacobian_blocks(func, t, y)
        
        # Create preconditioner blocks
        M_inv_blocks = []
        for i, J_block in enumerate(J_blocks):
            # Ensure J_block is in the correct dtype
            J_block = J_block.to(dtype=self.dtype, device=self.device)
            
            # Add regularization
            M = J_block + self.precond_regularization * torch.eye(J_block.shape[0], dtype=self.dtype, device=self.device)
            
            # Compute LU decomposition
            try:
                lu, pivots = torch.linalg.lu_factor(M)
                M_inv_blocks.append((lu, pivots))
            except RuntimeError:
                # If LU fails, use identity
                M_inv_blocks.append(None)
        
        def precon_solve(v):
            """Apply preconditioner to vector v."""
            v = v.to(dtype=self.dtype, device=self.device)
            out_flat = torch.zeros_like(v)
            
            for i, (start_idx, end_idx) in enumerate(zip(self.precond_block_sizes[:-1], self.precond_block_sizes[1:])):
                v_sub = v[start_idx:end_idx]
                if M_inv_blocks[i] is not None:
                    out_flat[start_idx:end_idx] = torch.linalg.lu_solve(
                        M_inv_blocks[i][0], 
                        M_inv_blocks[i][1], 
                        v_sub.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    out_flat[start_idx:end_idx] = v_sub
            
            return out_flat
        
        return precon_solve

    def _adaptive_gmres_tolerance(self, base_tol):
        if not self.gmres_adaptive or self.last_error_norm == 1.0: return base_tol
        eta = self.gmres_tol_current
        norm_ratio = self.last_error_norm / max(base_tol / eta, 1e-10)
        eta_new = 0.9 * (norm_ratio ** 0.5)
        eta_new = max(0.01 * eta, min(0.9, eta_new))
        if self.monitor_gradient_norm and len(self.gradient_norm_history) > 0:
            recent_grad_norm = self.gradient_norm_history[-1]
            if recent_grad_norm < 1e-6: eta_new *= 0.1
            elif recent_grad_norm > 1e3: eta_new *= 2.0
        if self.consecutive_rejects > 0: eta_new *= 0.5 ** self.consecutive_rejects
        self.gmres_tol_current = eta_new
        return eta_new * base_tol

# Example usage
if __name__ == '__main__':
    from torchdiffeq import odeint_adjoint
    
    class InverterController(nn.Module):
        def __init__(self, state_dim=3, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        def forward(self, state): return 400.0 * self.net(state)

    class StiffInverterSystem(nn.Module):
        def __init__(self, controller):
            super().__init__()
            self.register_buffer('L1', torch.tensor(0.5e-3))
            self.register_buffer('L2', torch.tensor(0.3e-3))
            self.register_buffer('C', torch.tensor(10e-6))
            self.register_buffer('R1', torch.tensor(0.01))
            self.register_buffer('R2', torch.tensor(0.02))
            self.controller = controller
        def forward(self, t, state):
            i_L1, i_L2, v_C = state[..., 0], state[..., 1], state[..., 2]
            v_in = self.controller(state).squeeze(-1)
            di_L1_dt = (v_in - v_C - self.R1 * i_L1) / self.L1
            di_L2_dt = (v_C - self.R2 * i_L2) / self.L2
            dv_C_dt = (i_L1 - i_L2) / self.C
            return torch.stack([di_L1_dt, di_L2_dt, dv_C_dt], dim=-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    controller = InverterController().to(device, dtype)
    dynamics = StiffInverterSystem(controller).to(device, dtype)
    
    y0 = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    t_span = torch.linspace(0, 0.02, 201, device=device, dtype=dtype)
    
    solver_options = {
        'rtol': 1e-7, 'atol': 1e-9, 'use_preconditioning': True,
        'precond_type': 'block_diagonal', 'precond_block_sizes': [3],
        'gradient_checkpointing': True, 'monitor_gradient_norm': True,
        'adaptive_regularization': True, 'gradient_warmup_steps': 100,
        'use_mixed_precision': False, 'jacobian_autodiff_threshold': 10,
    }
    
    print("Solving with Corrected EnhancedNeuralODESolver...")
    start_time = time.time()
    
    solution = odeint_adjoint(dynamics, y0, t_span, method=EnhancedNeuralODESolver, options=solver_options)
    
    elapsed = time.time() - start_time
    print(f"Solution completed in {elapsed:.3f} seconds.")
    
    print("\n--- Verifying Gradient Flow ---")
    target_voltage = torch.tensor(50.0, device=device, dtype=dtype)
    loss = (solution[-1, 2] - target_voltage)**2
    
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    
    has_grads = y0.grad is not None and y0.grad.norm().item() > 0
    print(f"Gradient norm for y0: {y0.grad.norm().item():.6e}" if has_grads else "y0: No gradient!")
    
    for name, param in controller.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            has_grads = True
            print(f"  - {name}: grad_norm = {param.grad.norm().item():.6e}")
        else:
            has_grads = False
            print(f"  - {name}: No gradient or zero gradient!")
    
    if has_grads:
        print("\n✅ SUCCESS: Gradient flow preserved through the corrected stiff solver!")
    else:
        print("\n❌ FAILURE: Gradient flow broken!")
