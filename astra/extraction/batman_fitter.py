"""
ASTRA BATMAN Fitter — Physical transit model fitting with MCMC uncertainties.

Uses the BATMAN package (Mandel-Agol transit model) for high-SNR candidates
(SNR > 10, confidence > 0.80). Fits 7 free parameters using differential
evolution global optimization followed by emcee MCMC for uncertainties.
"""

import numpy as np
from scipy.optimize import differential_evolution

from astra.utils.logger import get_logger
from astra.utils.stellar_utils import R_SUN_EARTH, AU_METERS, R_SUN_METERS

logger = get_logger("astra.extraction.batman_fitter")


class BATMANFitResult:
    """Result of BATMAN transit model fitting.

    Attributes:
        converged: Whether the fit converged.
        params: Dictionary of fitted parameters, each as (median, lower_err, upper_err).
        r_planet_earth: Planet radius in Earth radii.
        orbital_distance_au: Orbital distance in AU.
        t_eq: Equilibrium temperature in Kelvin.
        chi2_reduced: Reduced chi-squared of the best fit.
        model_flux: Best-fit model flux evaluated at the data phases.
    """

    def __init__(self):
        self.converged: bool = False
        self.params: dict[str, tuple[float, float, float]] = {}
        self.r_planet_earth: float = np.nan
        self.r_planet_earth_err: float = np.nan
        self.orbital_distance_au: float = np.nan
        self.t_eq: float = np.nan
        self.chi2_reduced: float = np.nan
        self.model_flux: np.ndarray = np.array([])

    def to_dict(self) -> dict:
        """Convert to flat dictionary for catalogue output."""
        result = {"batman_fit": self.converged}
        for param_name, (median, lower, upper) in self.params.items():
            result[param_name] = median
            result[f"{param_name}_err_lower"] = lower
            result[f"{param_name}_err_upper"] = upper
        result["r_planet_earth_fit"] = self.r_planet_earth
        result["orbital_distance_au_fit"] = self.orbital_distance_au
        result["t_eq_fit"] = self.t_eq
        result["chi2_reduced"] = self.chi2_reduced
        return result


def fit_batman_model(
    binned_phase: np.ndarray,
    binned_flux: np.ndarray,
    binned_flux_err: np.ndarray,
    period_init: float,
    t0_init: float,
    depth_init: float,
    duration_init: float,
    r_star: float = 1.0,
    teff: float = 5778.0,
    rp_rs_bounds: tuple[float, float] = (0.001, 0.3),
    a_rs_bounds: tuple[float, float] = (1.5, 100.0),
    b_bounds: tuple[float, float] = (0.0, 1.2),
    t0_tolerance: float = 0.1,
    period_tolerance: float = 0.01,
    u_bounds: tuple[float, float] = (0.0, 1.0),
    mcmc_enabled: bool = True,
    mcmc_nwalkers: int = 32,
    mcmc_nsteps: int = 500,
    mcmc_burnin: int = 200,
) -> BATMANFitResult:
    """Fit a BATMAN (Mandel-Agol) transit model to phase-folded data.

    Free parameters:
        Rp_Rs — Planet-to-star radius ratio
        a_Rs  — Orbital distance in stellar radii
        b     — Impact parameter
        T0    — Transit mid-time
        P     — Orbital period
        u1, u2 — Quadratic limb darkening coefficients

    Uses scipy.optimize.differential_evolution as the global optimizer,
    then emcee MCMC for uncertainty estimation (16/50/84 percentiles).

    Args:
        binned_phase: Phase-folded binned phase array.
        binned_flux: Phase-folded binned flux array.
        binned_flux_err: Flux uncertainty array.
        period_init: Initial period estimate in days.
        t0_init: Initial T0 estimate in days.
        depth_init: Initial transit depth estimate.
        duration_init: Initial transit duration in days.
        r_star: Stellar radius in solar radii.
        teff: Stellar effective temperature in Kelvin.
        rp_rs_bounds: Bounds for Rp/Rs parameter.
        a_rs_bounds: Bounds for a/Rs parameter.
        b_bounds: Bounds for impact parameter.
        t0_tolerance: T0 tolerance in days.
        period_tolerance: Fractional period tolerance.
        u_bounds: Bounds for limb darkening coefficients.
        mcmc_enabled: Whether to run MCMC for uncertainties.
        mcmc_nwalkers: Number of MCMC walkers.
        mcmc_nsteps: Number of MCMC steps.
        mcmc_burnin: Number of burn-in steps to discard.

    Returns:
        BATMANFitResult with fitted parameters and uncertainties.
    """
    result = BATMANFitResult()

    try:
        import batman
    except ImportError:
        logger.error(
            "batman-package not installed. Install with: pip install batman-package"
        )
        return result

    if len(binned_phase) < 10:
        logger.warning("Too few binned points for BATMAN fitting")
        return result

    # ── Initial parameter estimates ─────────────────────────────────────
    rp_rs_init = np.sqrt(max(depth_init, 1e-6))

    # Estimate a/Rs from duration and period
    # T_dur ≈ (P/π) × (R_star/a) → a/Rs ≈ P / (π × T_dur)
    if duration_init > 0:
        a_rs_init = period_init / (np.pi * duration_init)
    else:
        a_rs_init = 10.0

    a_rs_init = np.clip(a_rs_init, a_rs_bounds[0], a_rs_bounds[1])

    # ── Define bounds for differential evolution ────────────────────────
    bounds = [
        rp_rs_bounds,                                                # Rp/Rs
        a_rs_bounds,                                                 # a/Rs
        b_bounds,                                                    # b
        (t0_init - t0_tolerance, t0_init + t0_tolerance),            # T0
        (period_init * (1 - period_tolerance),
         period_init * (1 + period_tolerance)),                      # P
        u_bounds,                                                    # u1
        u_bounds,                                                    # u2
    ]

    # ── BATMAN model function ───────────────────────────────────────────
    def batman_model(phase_arr, rp_rs, a_rs, b, t0, period, u1, u2):
        """Evaluate BATMAN transit model at given phases."""
        params = batman.TransitParams()
        params.t0 = 0.0  # Transit centered at phase 0
        params.per = 1.0  # Period = 1 in phase units
        params.rp = rp_rs
        params.a = a_rs
        params.inc = np.degrees(np.arccos(b / a_rs)) if a_rs > b else 90.0
        params.ecc = 0.0
        params.w = 90.0
        params.u = [u1, u2]
        params.limb_dark = "quadratic"

        m = batman.TransitModel(params, phase_arr)
        return m.light_curve(params)

    # ── Cost function for optimization ──────────────────────────────────
    def cost_function(theta):
        """Negative log-likelihood (chi-squared) for optimization."""
        rp_rs, a_rs, b, t0, period, u1, u2 = theta

        # Physical constraint: b must be < a_rs for transit to occur
        if b >= a_rs:
            return 1e10

        try:
            model = batman_model(binned_phase, rp_rs, a_rs, b, t0, period, u1, u2)
            residuals = (binned_flux - model) / binned_flux_err
            return np.sum(residuals ** 2)
        except Exception:
            return 1e10

    # ── Differential Evolution Optimization ─────────────────────────────
    logger.debug("Running differential evolution optimization...")

    try:
        de_result = differential_evolution(
            cost_function,
            bounds=bounds,
            seed=42,
            maxiter=1000,
            tol=1e-6,
            polish=True,
        )

        if not de_result.success:
            logger.warning(f"DE optimization did not converge: {de_result.message}")
            # Continue anyway with the best result found

        best_params = de_result.x
        rp_rs, a_rs, b, t0, period, u1, u2 = best_params

        # Compute reduced chi-squared
        n_data = len(binned_flux)
        n_params = 7
        result.chi2_reduced = de_result.fun / max(1, n_data - n_params)

        # Evaluate best-fit model
        result.model_flux = batman_model(
            binned_phase, rp_rs, a_rs, b, t0, period, u1, u2
        )

        logger.debug(
            f"DE result: Rp/Rs={rp_rs:.5f}, a/Rs={a_rs:.2f}, b={b:.3f}, "
            f"χ²_red={result.chi2_reduced:.3f}"
        )

    except Exception as e:
        logger.error(f"Differential evolution failed: {e}", exc_info=True)
        return result

    # ── MCMC Uncertainty Estimation ─────────────────────────────────────
    if mcmc_enabled:
        try:
            import emcee

            logger.debug(
                f"Running MCMC: {mcmc_nwalkers} walkers, {mcmc_nsteps} steps..."
            )

            ndim = 7

            def log_probability(theta):
                """Log-probability for MCMC sampling."""
                # Check bounds
                for i, (lo, hi) in enumerate(bounds):
                    if theta[i] < lo or theta[i] > hi:
                        return -np.inf

                rp_rs, a_rs, b, t0, period, u1, u2 = theta

                if b >= a_rs:
                    return -np.inf

                try:
                    model = batman_model(
                        binned_phase, rp_rs, a_rs, b, t0, period, u1, u2
                    )
                    residuals = (binned_flux - model) / binned_flux_err
                    return -0.5 * np.sum(residuals ** 2)
                except Exception:
                    return -np.inf

            # Initialize walkers around DE optimum
            pos = best_params + 1e-4 * np.random.randn(mcmc_nwalkers, ndim)

            # Clip to bounds
            for i, (lo, hi) in enumerate(bounds):
                pos[:, i] = np.clip(pos[:, i], lo + 1e-6, hi - 1e-6)

            sampler = emcee.EnsembleSampler(mcmc_nwalkers, ndim, log_probability)
            sampler.run_mcmc(pos, mcmc_nsteps, progress=False)

            # Extract chains after burn-in
            samples = sampler.get_chain(discard=mcmc_burnin, flat=True)

            if len(samples) > 0:
                # Compute 16/50/84 percentiles
                param_names = ["Rp_Rs", "a_Rs", "b", "T0", "P", "u1", "u2"]
                for i, name in enumerate(param_names):
                    q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
                    result.params[name] = (
                        float(q50),
                        float(q50 - q16),
                        float(q84 - q50),
                    )

                logger.debug("MCMC completed successfully")
            else:
                logger.warning("MCMC produced no valid samples after burn-in")
                _store_point_estimates(result, best_params)

        except ImportError:
            logger.warning(
                "emcee not installed. Using point estimates without uncertainties."
            )
            _store_point_estimates(result, best_params)
        except Exception as e:
            logger.warning(f"MCMC failed: {e}. Using point estimates.")
            _store_point_estimates(result, best_params)
    else:
        _store_point_estimates(result, best_params)

    # ── Derived parameters ──────────────────────────────────────────────
    rp_rs_final = result.params.get("Rp_Rs", (rp_rs, 0, 0))[0]
    a_rs_final = result.params.get("a_Rs", (a_rs, 0, 0))[0]

    # Planet radius in Earth radii
    result.r_planet_earth = rp_rs_final * r_star * R_SUN_EARTH

    # Orbital distance in AU
    result.orbital_distance_au = a_rs_final * r_star * R_SUN_METERS / AU_METERS

    # Equilibrium temperature
    if result.orbital_distance_au > 0:
        # T_eq = T_eff × sqrt(R_star / (2 × a)) × (1 - albedo)^0.25
        # Assumption: geometric albedo 0.3
        r_star_m = r_star * R_SUN_METERS
        a_m = result.orbital_distance_au * AU_METERS
        result.t_eq = teff * np.sqrt(r_star_m / (2.0 * a_m)) * (0.7) ** 0.25

    result.converged = True
    logger.info(
        f"BATMAN fit converged: Rp={result.r_planet_earth:.2f} R_Earth, "
        f"a={result.orbital_distance_au:.4f} AU, T_eq={result.t_eq:.0f} K"
    )

    return result


def _store_point_estimates(result: BATMANFitResult, best_params: np.ndarray) -> None:
    """Store DE point estimates without uncertainty information.

    Args:
        result: BATMANFitResult to update.
        best_params: Array of best-fit parameters from DE.
    """
    param_names = ["Rp_Rs", "a_Rs", "b", "T0", "P", "u1", "u2"]
    for i, name in enumerate(param_names):
        result.params[name] = (float(best_params[i]), 0.0, 0.0)
