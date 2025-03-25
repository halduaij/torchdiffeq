import torch 
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver 

# Precompute constants to avoid redundant calculations
# This saves repeated tensor creation and square root computation
_SQRT_6 = torch.sqrt(torch.tensor(6., dtype=torch.float64))
_4_MINUS_SQRT_6 = 4 - _SQRT_6
_4_PLUS_SQRT_6 = 4 + _SQRT_6
_16_MINUS_SQRT_6 = 16 - _SQRT_6
_16_PLUS_SQRT_6 = 16 + _SQRT_6

# Radau IIA coefficients (order 5, 3 stages) with optimized calculations
# These calculations remain mathematically identical but avoid redundant computations
_RADAU_IIA_TABLEAU = _ButcherTableau( 
    alpha=torch.tensor([ 
        _4_MINUS_SQRT_6 / 10, 
        _4_PLUS_SQRT_6 / 10, 
        1. 
    ], dtype=torch.float64), 
    beta=[ 
        torch.tensor([_4_MINUS_SQRT_6 / 10], dtype=torch.float64), 
        torch.tensor([ 
            (88 - 7 * _SQRT_6) / 360, 
            (296 + 169 * _SQRT_6) / 1800 
        ], dtype=torch.float64), 
        torch.tensor([ 
            (296 - 169 * _SQRT_6) / 1800, 
            (88 + 7 * _SQRT_6) / 360, 
            _16_MINUS_SQRT_6 / 36 
        ], dtype=torch.float64), 
    ], 
    c_sol=torch.tensor([ 
        _16_MINUS_SQRT_6 / 36, 
        _16_PLUS_SQRT_6 / 36, 
        1/9, 
        0. 
    ], dtype=torch.float64), 
    c_error=torch.tensor([ 
        _16_MINUS_SQRT_6 / 36 - (1/9), 
        _16_PLUS_SQRT_6 / 36 - (1/9), 
        0., 
        0. 
    ], dtype=torch.float64), 
) 

# Interpolation coefficients for dense output - precomputed
# Mathematically identical but computationally more efficient
RADAU_C_MID = torch.tensor([ 
    0.5 * (_16_MINUS_SQRT_6 / 36), 
    0.5 * (_16_PLUS_SQRT_6 / 36), 
    0.5 * (1/9), 
    0. 
], dtype=torch.float64)

class RadauSolver(RKAdaptiveStepsizeODESolver): 
    order = 5 
    tableau = _RADAU_IIA_TABLEAU 
    mid = RADAU_C_MID