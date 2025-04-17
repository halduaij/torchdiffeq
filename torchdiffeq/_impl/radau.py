# ---------------------------------------------------------------------------
#  radau.py  – explicit Radau coefficients (unchanged) + NEW implicit DIRK
# ---------------------------------------------------------------------------
import math
import torch
from torch import Tensor

# ── 0.  EXPLICIT RADAU‑IIA TABLEAU  (kept from your original file) ───────────
_SQRT_6 = math.sqrt(6.0)

_4_MINUS_SQRT_6  = 4 - _SQRT_6
_4_PLUS_SQRT_6   = 4 + _SQRT_6
_16_MINUS_SQRT_6 = 16 - _SQRT_6
_16_PLUS_SQRT_6  = 16 + _SQRT_6

from .rk_common import _ButcherTableau          # original import

_RADAU_IIA_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([_4_MINUS_SQRT_6 / 10,
                        _4_PLUS_SQRT_6 / 10,
                        1.], dtype=torch.float64),
    beta=[
        torch.tensor([_4_MINUS_SQRT_6 / 10], dtype=torch.float64),
        torch.tensor([(88 - 7 * _SQRT_6)  / 360,
                      (296 + 169 * _SQRT_6) / 1800], dtype=torch.float64),
        torch.tensor([(296 - 169 * _SQRT_6) / 1800,
                      (88  +   7 * _SQRT_6) / 360,
                       _16_MINUS_SQRT_6 / 36], dtype=torch.float64),
    ],
    c_sol=torch.tensor([_16_MINUS_SQRT_6 / 36,
                        _16_PLUS_SQRT_6 / 36,
                        1/9,
                        0.], dtype=torch.float64),
    c_error=torch.tensor([_16_MINUS_SQRT_6 / 36 - 1/9,
                          _16_PLUS_SQRT_6 / 36 - 1/9,
                          0., 0.], dtype=torch.float64),
)

# Mid‑point weights for dense output (unchanged)
RADAU_C_MID = torch.tensor([0.5 * (_16_MINUS_SQRT_6 / 36),
                            0.5 * (_16_PLUS_SQRT_6 / 36),
                            0.5 / 9,
                            0.], dtype=torch.float64)

# ── 1.  CONSTANTS FOR THE IMPLICIT DIRK SOLVER  ──────────────────────────────
# lower‑triangular Radau‑IIA(5) A‑matrix
_RADAU_A = torch.tensor([
    [(88-  7*_SQRT_6)/360,       0.,                 0.],
    [(296-169*_SQRT_6)/1800, (88+7*_SQRT_6)/360,     0.],
    [(16-  6*_SQRT_6)/36,   (16+6*_SQRT_6)/36,     1/9],
], dtype=torch.float64)

# weights, nodes, local error coefficients
_RADAU_B = torch.tensor([(16-6*_SQRT_6)/36,
                         (16+6*_SQRT_6)/36,
                          1/9], dtype=torch.float64)

_RADAU_C = torch.tensor([ (4-_SQRT_6)/10,
                          (4+_SQRT_6)/10,
                           1.0], dtype=torch.float64)

_RADAU_E = _RADAU_B - torch.tensor([1/9, 1/9, 0.0], dtype=torch.float64)

__all__ = [
    "_RADAU_IIA_TABLEAU", "RADAU_C_MID",
    "_RADAU_A", "_RADAU_B", "_RADAU_C", "_RADAU_E",
    "_radau5_implicit_dirk",
]

# ── 2.  IMPLICIT DIRK RADAU‑IIA(5) STEP  (gradient‑exact)  ───────────────────
def _radau5_implicit_dirk(
        func,
        t0: Tensor,
        y0: Tensor,
        f0: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        E: Tensor,
        newton_tol: float = 1e-9,
        max_newton: int   = 6,
    ):
    """
    One implicit Radau‑IIA(5) step solved stage‑by‑stage (DIRK style).

    *Per‑stage* Jacobians are built with `create_graph=True`, so gradients
    flow through Newton iterations and back into model parameters.

    Returns
    -------
    y1 : Tensor            – state at t0+dt
    f1 : Tensor            – derivative at t0+dt
    err: Tensor            – error estimate
    K  : Tensor (3,n)      – stage derivatives (for dense interpolation)
    """
    device, dtype = y0.device, y0.dtype
    s, n = 3, y0.numel()

    A = A.to(dtype=dtype, device=device)
    B = B.to(dtype=dtype, device=device)
    C = C.to(dtype=dtype, device=device)
    E = E.to(dtype=dtype, device=device)

    # Initial guess = explicit Euler extrapolation (all stages = f0)
    K = torch.stack([f0] * s, dim=0).requires_grad_(True)

    I_n = torch.eye(n, dtype=dtype, device=device)

    for _ in range(max_newton):
        max_update = torch.tensor(0., dtype=dtype, device=device)

        # Sequential DIRK solve (lower‑triangular A)
        for i in range(s):
            # 2.1  Stage state
            Yi = y0 + dt * torch.sum(A[i, :i+1].unsqueeze(-1) * K[:i+1], dim=0)

            # 2.2  Residual
            fi = func(t0 + C[i] * dt, Yi.view_as(y0))
            Ri = K[i] - fi                                # (n,)

            # 2.3  Jacobian J_i = d f_i / d y_i
            Ji = torch.autograd.functional.jacobian(
                    lambda y: func(t0 + C[i]*dt, y),
                    Yi.view_as(y0),
                    create_graph=True).reshape(n, n)

            # 2.4  Solve (I - dt*Aii*Ji) Δ = -Ri
            M   = I_n - dt * A[i, i] * Ji
            dKi = torch.linalg.solve(M, -Ri)

            # 2.5  Update stage derivative
            K[i] = K[i] + dKi
            max_update = torch.maximum(max_update, dKi.norm())

        if max_update < newton_tol:
            break

    # 3.  Produce step result + error estimate
    y1  = y0 + dt * torch.sum(B.unsqueeze(-1) * K, dim=0)
    f1  = func(t0 + dt, y1.view_as(y0))
    err = dt * torch.sum(E.unsqueeze(-1) * K, dim=0)
    return y1.view_as(y0), f1, err.view_as(y0), K
