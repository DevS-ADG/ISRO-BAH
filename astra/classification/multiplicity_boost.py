"""
ASTRA Multiplicity Boost — Bayesian posterior update for multi-planet systems.

In planetary systems, the probability that multiple signals around the same
star are all false positives is astronomically small. This is the validated
Lissauer et al. / Rowe et al. multiplicity argument from NASA's Kepler pipeline.

Implementation: For each marginal candidate on a multi-candidate star,
compute a Bayesian posterior update where the joint FP probability scales
as approximately FP_prior^N.
"""

import numpy as np

from astra.utils.logger import get_logger

logger = get_logger("astra.classification.multiplicity_boost")


def apply_multiplicity_boost(
    candidates: list[dict],
    min_planet_prob: float = 0.50,
    max_boost_cap: float = 0.95,
) -> list[dict]:
    """Apply Bayesian multiplicity boost to multi-planet candidates.

    Trigger: A star must have 2+ candidates each with P(PLANET) > min_planet_prob
    before the boost.

    Logic: For N candidates on a star, the joint false positive probability
    scales as FP_prior^N, making it exponentially unlikely that all signals
    are false positives.

    Expected behavior: A candidate with 60% planet confidence on a
    multi-planet star should boost to ~85% confidence.

    Hard cap: Cannot raise above max_boost_cap (0.95). Cannot boost
    candidates that were hard-rejected by vetting.

    Args:
        candidates: List of candidate dictionaries. Each must have:
            - tic_id: Star identifier
            - confidence_planet: P(PLANET)
            - probabilities: Full probability vector [P, EB, BLEND, OTHER]
            - vetting_passed: Boolean
        min_planet_prob: Minimum P(PLANET) to consider for boost.
        max_boost_cap: Maximum boosted confidence (default 0.95).

    Returns:
        Updated list of candidates with boosted probabilities where applicable.
    """
    if not candidates:
        return candidates

    # Group candidates by star (TIC_ID)
    star_groups: dict[int, list[int]] = {}
    for i, cand in enumerate(candidates):
        tic_id = cand.get("tic_id", 0)
        if tic_id not in star_groups:
            star_groups[tic_id] = []
        star_groups[tic_id].append(i)

    n_boosted = 0

    for tic_id, indices in star_groups.items():
        # Check if this star qualifies for multiplicity boost
        qualifying = [
            i for i in indices
            if candidates[i].get("confidence_planet", 0) >= min_planet_prob
            and candidates[i].get("vetting_passed", True)
        ]

        if len(qualifying) < 2:
            continue  # Need at least 2 candidates above threshold

        n_candidates = len(qualifying)
        logger.info(
            f"TIC {tic_id}: {n_candidates} qualifying candidates — "
            f"applying multiplicity boost"
        )

        for idx in qualifying:
            cand = candidates[idx]

            # Skip hard-rejected candidates
            if not cand.get("vetting_passed", True):
                continue

            old_p_planet = cand["confidence_planet"]

            # Bayesian update: FP_prior → FP_posterior
            fp_prior = 1.0 - old_p_planet
            fp_posterior = fp_prior ** n_candidates

            # New planet probability
            p_planet_new = 1.0 - fp_posterior

            # Apply hard cap
            p_planet_new = min(p_planet_new, max_boost_cap)

            # Only boost, never decrease
            if p_planet_new <= old_p_planet:
                continue

            # Update probability vector, scaling non-planet classes
            if "probabilities" in cand:
                old_proba = np.array(cand["probabilities"])
                new_proba = old_proba.copy()
                new_proba[0] = p_planet_new

                # Redistribute remaining probability among other classes
                remaining = 1.0 - p_planet_new
                old_non_planet = np.sum(old_proba[1:])

                if old_non_planet > 0:
                    for j in range(1, 4):
                        new_proba[j] = old_proba[j] / old_non_planet * remaining
                else:
                    new_proba[1:] = remaining / 3.0

                cand["probabilities"] = new_proba
                cand["confidence_eb"] = float(new_proba[1])
                cand["confidence_blend"] = float(new_proba[2])
                cand["confidence_other"] = float(new_proba[3])

            cand["confidence_planet"] = float(p_planet_new)
            cand["multiplicity_boost_applied"] = True

            n_boosted += 1
            logger.info(
                f"  Candidate {cand.get('candidate_number', '?')}: "
                f"P(PLANET) {old_p_planet:.3f} → {p_planet_new:.3f} "
                f"(N={n_candidates}, FP: {fp_prior:.3f} → {fp_posterior:.5f})"
            )

    logger.info(f"Multiplicity boost: {n_boosted} candidates boosted")
    return candidates
