"""
ASTRA Multi-Planet Search — Iterative transit detection with masking.

Implements the iterative BLS/TLS multi-planet search algorithm:
1. Run BLS+TLS on full light curve.
2. If SNR > threshold, record as TCE candidate, mask in-transit data.
3. Repeat on masked light curve up to max_planets_per_star times.

SNR gating:
  SNR < 5:     Discard immediately. Log to noise file.
  5 ≤ SNR < 7: Flag as weak/marginal candidate. Skip vetting/ML.
  SNR ≥ 7:     Proceed to extraction and vetting.
  SNR ≥ 10:    Full BATMAN model fitting in addition to standard processing.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from astra.detection.bls_search import BLSResult, run_bls_search
from astra.detection.tls_search import TLSResult, run_tls_search
from astra.utils.logger import get_logger
from astra.utils.stellar_utils import compute_snr

logger = get_logger("astra.detection.multi_planet")


@dataclass
class TCERecord:
    """Threshold Crossing Event record for a single candidate.

    Contains all detection-level information for a single transit signal
    candidate found around a star.
    """

    tic_id: int = 0
    candidate_number: int = 0
    period: float = np.nan
    t0: float = np.nan
    duration: float = np.nan       # In days
    depth: float = np.nan          # Fractional flux decrease
    snr: float = 0.0
    n_transits: int = 0
    bls_power: float = np.nan
    tls_sde: float = np.nan
    bls_fap: float = 1.0
    transit_times: np.ndarray = field(default_factory=lambda: np.array([]))
    tier: str = "unvetted"          # marginal_unvetted, unvetted, etc.
    snr_category: str = "noise"     # noise, weak, standard, high

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "tic_id": self.tic_id,
            "candidate_number": self.candidate_number,
            "period": self.period,
            "t0": self.t0,
            "duration_days": self.duration,
            "duration_hours": self.duration * 24.0,
            "depth": self.depth,
            "depth_ppm": self.depth * 1e6,
            "snr": self.snr,
            "n_transits": self.n_transits,
            "bls_power": self.bls_power,
            "tls_sde": self.tls_sde,
            "bls_fap": self.bls_fap,
            "tier": self.tier,
            "snr_category": self.snr_category,
        }


def search_multi_planet(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    tic_id: int,
    r_star: float = 1.0,
    m_star: float = 1.0,
    teff: float = 5778.0,
    period_min: float = 0.5,
    period_max: float = 13.0,
    snr_threshold: float = 7.0,
    snr_discard_threshold: float = 5.0,
    n_trial_periods: int = 19500,
    max_planets: int = 5,
) -> list[TCERecord]:
    """Run iterative multi-planet transit search.

    Algorithm:
    1. Run BLS and TLS on the full detrended light curve.
    2. Combine results, taking the detection with higher SNR.
    3. If SNR > snr_threshold, record as TCE and mask in-transit data.
    4. Repeat on masked light curve until no signal found or max reached.

    Args:
        time: Detrended time array in days.
        flux: Detrended normalized flux array.
        flux_err: Flux uncertainty array.
        tic_id: TIC ID of the star.
        r_star: Stellar radius in solar radii.
        m_star: Stellar mass in solar masses.
        teff: Stellar effective temperature in Kelvin.
        period_min: Minimum search period in days.
        period_max: Maximum search period in days.
        snr_threshold: Minimum SNR to record a TCE (default 7.0).
        snr_discard_threshold: SNR below this is discarded (default 5.0).
        n_trial_periods: Number of trial periods for BLS.
        max_planets: Maximum candidates per star (default 5).

    Returns:
        List of TCERecord objects for all detected candidates.
    """
    tces: list[TCERecord] = []
    current_time = time.copy()
    current_flux = flux.copy()
    current_flux_err = flux_err.copy()

    for candidate_num in range(1, max_planets + 1):
        if len(current_time) < 50:
            logger.debug(
                f"TIC {tic_id}: Too few points remaining ({len(current_time)}) "
                f"after {candidate_num - 1} candidates"
            )
            break

        # ── Run BLS ─────────────────────────────────────────────────────
        bls_result = run_bls_search(
            current_time,
            current_flux,
            current_flux_err,
            period_min=period_min,
            period_max=period_max,
            n_trial_periods=n_trial_periods,
        )

        # ── Run TLS ─────────────────────────────────────────────────────
        tls_result = run_tls_search(
            current_time,
            current_flux,
            current_flux_err,
            r_star=r_star,
            m_star=m_star,
            teff=teff,
            period_min=period_min,
            period_max=period_max,
        )

        # ── Combine BLS + TLS: take the detection with higher SNR ──────
        tce = _combine_bls_tls(bls_result, tls_result, tic_id, candidate_num)

        # ── Physical sanity check on duration vs period ────────────────
        # A real planetary transit lasts 1-15% of the orbital period.
        # If duration > 25% of period, this is almost certainly a
        # spurious detection where the algorithm fit the entire orbit
        # as one transit box. Reject and stop searching.
        if tce.duration > 0 and tce.period > 0:
            duty_cycle = tce.duration / tce.period
            if duty_cycle > 0.25:
                logger.warning(
                    f"TIC {tic_id} candidate {candidate_num}: "
                    f"duration/period={duty_cycle:.2f} > 0.25 "
                    f"(duration={tce.duration*24:.1f}h, P={tce.period:.4f}d) "
                    f"-> REJECTED as unphysical"
                )
                break

        # Recompute SNR using the formula from spec
        in_transit_mask = _get_transit_mask(
            current_time, tce.period, tce.t0, tce.duration
        )
        n_in_transit = int(np.sum(in_transit_mask))
        rms_oot = np.nanstd(current_flux[~in_transit_mask]) if np.any(~in_transit_mask) else 1.0

        if tce.depth > 0 and n_in_transit > 0 and rms_oot > 0:
            tce.snr = compute_snr(tce.depth, n_in_transit, rms_oot)

        # ── SNR gating ─────────────────────────────────────────────────
        if tce.snr < snr_discard_threshold:
            # SNR < 5: Discard immediately
            tce.snr_category = "noise"
            logger.debug(
                f"TIC {tic_id} candidate {candidate_num}: "
                f"SNR={tce.snr:.1f} < {snr_discard_threshold} → DISCARDED"
            )
            break

        elif tce.snr < snr_threshold:
            # 5 ≤ SNR < 7: Weak/marginal candidate
            tce.snr_category = "weak"
            tce.tier = "marginal_unvetted"
            tces.append(tce)
            logger.debug(
                f"TIC {tic_id} candidate {candidate_num}: "
                f"SNR={tce.snr:.1f} → MARGINAL (no vetting/ML)"
            )
            break  # Don't search for more planets after a marginal detection

        else:
            # SNR ≥ 7: Standard candidate
            if tce.snr >= 10.0:
                tce.snr_category = "high"
            else:
                tce.snr_category = "standard"
            tce.tier = "unvetted"
            tces.append(tce)

            logger.info(
                f"TIC {tic_id} candidate {candidate_num}: "
                f"P={tce.period:.4f}d, depth={tce.depth:.5f}, "
                f"SNR={tce.snr:.1f}, n_transits={tce.n_transits} → TCE"
            )

            # ── Mask in-transit data for next iteration ─────────────────
            mask_out = ~_get_transit_mask(
                current_time, tce.period, tce.t0, tce.duration
            )
            current_time = current_time[mask_out]
            current_flux = current_flux[mask_out]
            current_flux_err = current_flux_err[mask_out]

    if not tces:
        logger.debug(f"TIC {tic_id}: No TCEs found above SNR threshold")

    return tces


def _combine_bls_tls(
    bls: BLSResult,
    tls: TLSResult,
    tic_id: int,
    candidate_num: int,
) -> TCERecord:
    """Combine BLS and TLS results, preferring TLS when valid.

    TLS uses a physically realistic limb-darkened transit template
    (Mandel-Agol), which produces more accurate period/depth/duration
    estimates than BLS's simple box model. TLS is preferred whenever
    it produces a valid detection (SDE > 5). BLS is used as fallback
    when TLS fails or produces no signal.

    BLS can produce artificially inflated SNR values when its box
    model fits long-duration "transits" that aren't physical. TLS
    is more robust against this because its template constrains the
    transit shape.

    Args:
        bls: BLS search result.
        tls: TLS search result.
        tic_id: TIC ID of the star.
        candidate_num: Candidate number (1-indexed).

    Returns:
        Combined TCERecord.
    """
    tce = TCERecord()
    tce.tic_id = tic_id
    tce.candidate_number = candidate_num
    tce.bls_power = bls.power
    tce.tls_sde = tls.sde
    tce.bls_fap = bls.fap

    bls_snr = bls.snr if np.isfinite(bls.snr) else 0.0
    tls_snr = tls.snr if np.isfinite(tls.snr) else 0.0
    tls_valid = np.isfinite(tls.period) and tls.sde > 5.0

    if tls_valid:
        # Prefer TLS — it uses physically realistic transit templates
        tce.period = tls.period
        tce.t0 = tls.t0
        tce.duration = tls.duration
        tce.depth = tls.depth
        tce.snr = tls_snr
        tce.n_transits = tls.n_transits
        tce.transit_times = tls.transit_times
        logger.debug(
            f"TIC {tic_id} cand {candidate_num}: using TLS "
            f"(SDE={tls.sde:.1f}, P={tls.period:.4f}d)"
        )
    elif np.isfinite(bls.period):
        # Fall back to BLS when TLS failed or produced no signal
        tce.period = bls.period
        tce.t0 = bls.t0
        tce.duration = bls.duration
        tce.depth = bls.depth
        tce.snr = bls_snr
        tce.n_transits = bls.n_transits
        if bls.n_transits > 0:
            tce.transit_times = bls.t0 + bls.period * np.arange(bls.n_transits)
        logger.debug(
            f"TIC {tic_id} cand {candidate_num}: using BLS fallback "
            f"(TLS SDE={tls.sde:.1f}, BLS SNR={bls_snr:.1f})"
        )
    else:
        # Neither search found anything
        tce.snr = 0.0

    return tce


def _get_transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> np.ndarray:
    """Create boolean mask for in-transit data points.

    Masks ±0.5 × duration around each predicted transit time.

    Args:
        time: Time array.
        period: Orbital period in days.
        t0: Transit mid-time.
        duration: Transit duration in days.

    Returns:
        Boolean mask (True = in transit).
    """
    if np.isnan(period) or np.isnan(t0) or np.isnan(duration) or period <= 0:
        return np.zeros(len(time), dtype=bool)

    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    half_dur_phase = (duration / period) / 2.0
    return np.abs(phase) <= half_dur_phase
