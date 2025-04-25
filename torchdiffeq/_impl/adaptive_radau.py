# adaptive_radau.py – adaptive, fully-implicit Radau-IIA (orders-3 & 5)
import torch
from .rk_common import (_RungeKuttaState as _RKState, RKAdaptiveStepsizeODESolver,
                        _runge_kutta_step, _compute_error_ratio, _interp_fit,
                        _ButcherTableau)
from .fixed_grid_implicit import RadauIIA3 as _Radau3, RadauIIA5 as _Radau5
from .misc import Perturb


def _half_mid(c_sol: torch.Tensor) -> torch.Tensor:
    """mid-point weights for dense output"""
    return 0.5 * c_sol


# ───────────────────────────────────────────────────────────────────────
# generic adaptive wrapper
# ───────────────────────────────────────────────────────────────────────
class _Base(RKAdaptiveStepsizeODESolver):
    _inner_cls = None          # set by subclass
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, y0, rtol, atol, **kw):
        super().__init__(func, y0, rtol, atol, **kw)
        self._inner = self._inner_cls(func=func, y0=y0,
                                      step_size=None,
                                      atol=atol, rtol=rtol,
                                      max_iters=kw.get("max_iters", 100))

    # ------------------------------------------------------------------
    # one adaptive Radau step
    # ------------------------------------------------------------------
    def _adaptive_step(self, st: _RKState) -> _RKState:
        y0, f0, _, t0, dt, coeff = st
        dt = dt.clamp(self.min_step, self.max_step)
        self.func.callback_step(t0, y0, dt)
        t1 = t0 + dt

        # (1) explicit predictor (order p-1, cheap)
        y_pred, f_pred, _, k_pred = _runge_kutta_step(
            self.func, y0, f0, t0, dt, t1, tableau=self.tableau)

        # (2) implicit corrector (fully stiff-stable)
        dy_imp, _ = self._inner._step_func(self.func, t0, dt, t1, y0)
        y1 = y0 + dy_imp
        f1 = self.func(t1, y1, perturb=Perturb.PREV)

        # (3) error estimate = predictor – corrector
        err_vec = y1 - y_pred
        ratio = _compute_error_ratio(err_vec, self.rtol, self.atol, y0, y1, self.norm)

        # fast-accept if predictor already very good
        if ratio <= 0.05:
            self.func.callback_accept_step(t0, y0, dt)
            dt_next = dt * torch.clamp(
                0.9 * ratio.pow(-1 / (self.order + 1)), self.dfactor, self.ifactor
            ).clamp(self.min_step, self.max_step)
            coeff = self._interp_fit(y0, y_pred, k_pred, dt)
            return _RKState(y_pred, f_pred, t0, t1, dt_next, coeff)

        # normal accept/reject
        if (ratio <= 1) or (dt <= self.min_step):        # accept
            self.func.callback_accept_step(t0, y0, dt)
            coeff = self._interp_fit(y0, y1, k_pred, dt)
            dt_next = dt * torch.clamp(
                0.9 * ratio.pow(-1 / (self.order + 1)), self.dfactor, self.ifactor
            ).clamp(self.min_step, self.max_step)
            return _RKState(y1, f1, t0, t1, dt_next, coeff)

        # rejected step – shrink dt and retry
        self.func.callback_reject_step(t0, y0, dt)
        dt_next = dt * torch.clamp(
            0.9 * ratio.pow(-1 / (self.order + 1)), 0.1, 0.5
        ).clamp(self.min_step, self.max_step)
        return _RKState(y0, f0, t0, t0, dt_next, coeff)


# ───────────────────────────────────────────────────────────────────────
# Radau IIA – order-3 (2 stages)
# ───────────────────────────────────────────────────────────────────────
class AdaptiveRadauIIA3(_Base):
    order = 3
    tableau = _ButcherTableau(
        alpha=_Radau3.tableau.alpha,
        beta=[
            _Radau3.tableau.beta[0][:1].clone(),    # 1 entry
            _Radau3.tableau.beta[1][:2].clone(),    # 2 entries
        ],
        # add trailing zero so length == stages+1 (needed by predictor code)
        c_sol=torch.cat([_Radau3.tableau.c_sol,
                         torch.tensor([0.], dtype=_Radau3.tableau.c_sol.dtype)]),
        c_error=torch.cat([_Radau3.tableau.c_error,
                           torch.tensor([0.], dtype=_Radau3.tableau.c_error.dtype)]),
    )
    mid = _half_mid(tableau.c_sol)
    _inner_cls = _Radau3


# ───────────────────────────────────────────────────────────────────────
# Radau IIA – order-5 (3 stages)
# ───────────────────────────────────────────────────────────────────────
class AdaptiveRadauIIA5(_Base):
    order = 5
    tableau = _ButcherTableau(
        alpha=_Radau5.tableau.alpha,
        beta=[
            _Radau5.tableau.beta[0][:1].clone(),    # 1 entry
            _Radau5.tableau.beta[1][:2].clone(),    # 2 entries
            _Radau5.tableau.beta[2][:3].clone(),    # 3 entries
        ],
        c_sol=torch.cat([_Radau5.tableau.c_sol,
                         torch.tensor([0.], dtype=_Radau5.tableau.c_sol.dtype)]),
        c_error=torch.cat([_Radau5.tableau.c_error,
                           torch.tensor([0.], dtype=_Radau5.tableau.c_error.dtype)]),
    )
    mid = _half_mid(tableau.c_sol)
    _inner_cls = _Radau5
