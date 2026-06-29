"""
ASTRA Unit Tests — Core module tests with synthetic data.

Tests each critical module independently without requiring real TESS data.
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# STELLAR UTILITIES TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStellarUtils:
    """Tests for astra.utils.stellar_utils."""

    def test_compute_snr(self):
        """Test SNR computation: depth × sqrt(N) / RMS."""
        from astra.utils.stellar_utils import compute_snr

        # Known values: depth=0.01, N=100, RMS=0.001
        # SNR = 0.01 * sqrt(100) / 0.001 = 0.01 * 10 / 0.001 = 100
        snr = compute_snr(0.01, 100, 0.001)
        assert abs(snr - 100.0) < 0.01

    def test_compute_snr_zero_inputs(self):
        """Test SNR returns 0 for invalid inputs."""
        from astra.utils.stellar_utils import compute_snr

        assert compute_snr(0, 100, 0.001) == 0.0
        assert compute_snr(0.01, 0, 0.001) == 0.0
        assert compute_snr(0.01, 100, 0) == 0.0

    def test_planet_radius_earth(self):
        """Test planet radius computation from depth and stellar radius."""
        from astra.utils.stellar_utils import planet_radius_earth, R_SUN_EARTH

        # Jupiter-like: depth=0.01, R_star=1.0 R_sun
        # Rp = 1.0 * sqrt(0.01) * 109.076 = 10.9 R_earth
        rp = planet_radius_earth(0.01, 1.0)
        assert 10.0 < rp < 12.0  # ~10.9 R_earth

    def test_semi_major_axis(self):
        """Test Kepler's third law computation."""
        from astra.utils.stellar_utils import semi_major_axis_au

        # Earth: P=365.25 days, M=1.0 M_sun → a ≈ 1.0 AU
        a = semi_major_axis_au(365.25, 1.0)
        assert abs(a - 1.0) < 0.01

    def test_equilibrium_temperature(self):
        """Test equilibrium temperature computation."""
        from astra.utils.stellar_utils import equilibrium_temperature

        # Earth-like: Teff=5778, R_star=1.0, a=1.0 AU, albedo=0.3
        # T_eq ≈ 255 K
        t_eq = equilibrium_temperature(5778.0, 1.0, 1.0, albedo=0.3)
        assert 240 < t_eq < 270

    def test_estimate_stellar_mass_solar(self):
        """Test stellar mass estimation for Sun-like star."""
        from astra.utils.stellar_utils import estimate_stellar_mass

        m = estimate_stellar_mass(5778.0, 1.0)
        assert 0.8 < m < 1.2  # Should be near 1.0 M_sun


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalization:
    """Tests for astra.preprocessing.normalization."""

    def test_normalize_flux_median(self):
        """Test median normalization produces baseline ~1.0."""
        from astra.preprocessing.normalization import normalize_flux

        flux = np.array([1000.0, 1001.0, 999.0, 1000.5, 999.5])
        norm, _, factor = normalize_flux(flux, method="median")

        assert abs(np.median(norm) - 1.0) < 0.001
        assert abs(factor - 1000.0) < 1.0

    def test_sigma_clip(self):
        """Test sigma clipping removes outliers."""
        from astra.preprocessing.normalization import sigma_clip

        flux = np.ones(100)
        flux[50] = 100.0  # Extreme outlier

        mask = sigma_clip(flux, sigma=3.0)
        assert mask[50] is np.False_  # Outlier should be clipped
        assert np.sum(mask) == 99


# ═══════════════════════════════════════════════════════════════════════════
# PHASE FOLDING TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseFold:
    """Tests for astra.extraction.phase_fold."""

    def test_phase_fold_centering(self):
        """Test transit is centered at phase 0 after folding."""
        from astra.extraction.phase_fold import phase_fold

        # Create synthetic periodic signal
        period = 3.0
        t0 = 1.0
        duration = 0.1
        time = np.linspace(0, 27, 10000)
        flux = np.ones_like(time)

        # Add transits
        for i in range(10):
            t_mid = t0 + i * period
            mask = np.abs(time - t_mid) < duration / 2
            flux[mask] = 0.99  # 1% dip

        flux_err = np.ones_like(flux) * 0.001

        result = phase_fold(time, flux, flux_err, period, t0, duration)

        # Transit should be at phase ~0
        in_transit = np.abs(result.phase) < 0.02
        assert np.mean(result.flux[in_transit]) < 0.998

    def test_resample_fixed_length(self):
        """Test resampling to fixed length for CNN input."""
        from astra.extraction.phase_fold import resample_phase_fold

        phase = np.linspace(-0.5, 0.5, 500)
        flux = np.ones(500)
        flux[240:260] = 0.99

        resampled = resample_phase_fold(phase, flux, target_length=256)

        assert resampled.shape == (256,)
        assert abs(np.mean(resampled)) < 0.01  # Zero mean after normalization


# ═══════════════════════════════════════════════════════════════════════════
# GAP HANDLER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestGapHandler:
    """Tests for astra.preprocessing.gap_handler."""

    def test_find_gaps(self):
        """Test gap detection identifies correct segments."""
        from astra.preprocessing.gap_handler import find_gaps

        # Two segments with a 1-day gap
        time = np.concatenate([
            np.linspace(0, 10, 500),
            np.linspace(11, 20, 500),
        ])

        segments = find_gaps(time, gap_threshold_days=0.5)
        assert len(segments) == 2

    def test_no_gaps(self):
        """Test continuous data produces single segment."""
        from astra.preprocessing.gap_handler import find_gaps

        time = np.linspace(0, 27, 1000)
        segments = find_gaps(time, gap_threshold_days=0.5)
        assert len(segments) == 1


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureExtractor:
    """Tests for astra.extraction.feature_extractor."""

    def test_feature_count(self):
        """Test exactly 19 features are produced."""
        from astra.extraction.feature_extractor import FEATURE_NAMES

        assert len(FEATURE_NAMES) == 19

    def test_missing_features_are_nan(self):
        """Test missing features default to NaN, not zero."""
        from astra.extraction.feature_extractor import extract_features

        # Minimal input to trigger NaN defaults
        features = extract_features(
            phase=np.array([0.0]),
            flux=np.array([1.0]),
            flux_err=np.array([0.001]),
            binned_phase=np.array([0.0]),
            binned_flux=np.array([1.0]),
            period=3.0,
            t0=0.0,
            duration_days=0.1,
            depth=0.01,
            snr=10.0,
            n_transit=3,
            r_star=np.nan,  # Missing
            teff=np.nan,    # Missing
            crowdsap=np.nan,
        )

        # r_star and teff should be NaN, not zero
        assert np.isnan(features["r_star"])
        assert np.isnan(features["t_eff"])
        assert len(features) == 19


# ═══════════════════════════════════════════════════════════════════════════
# VETTING TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestOddEvenVetting:
    """Tests for astra.vetting.odd_even."""

    def test_identical_depths_pass(self):
        """Identical odd/even depths should pass the test."""
        from astra.vetting.odd_even import test_odd_even

        time = np.linspace(0, 27, 10000)
        flux = np.ones_like(time)

        # All transits have same depth
        period = 3.0
        t0 = 1.0
        duration = 0.1
        transit_times = t0 + period * np.arange(9)

        for t_mid in transit_times:
            mask = np.abs(time - t_mid) < duration / 2
            flux[mask] = 0.99

        flag, sigma, hard = test_odd_even(
            time, flux, transit_times, period, duration, sigma_threshold=3.0
        )

        assert not hard  # Should pass

    def test_alternating_depths_fail(self):
        """Alternating deep/shallow transits should trigger EB flag."""
        from astra.vetting.odd_even import test_odd_even

        time = np.linspace(0, 27, 50000)
        np.random.seed(42)
        flux = np.ones_like(time) + np.random.normal(0, 1e-4, size=len(time))

        period = 3.0
        t0 = 1.0
        duration = 0.1
        transit_times = t0 + period * np.arange(9)

        # Alternating depths: odd=deep, even=shallow
        for i, t_mid in enumerate(transit_times):
            mask = np.abs(time - t_mid) < duration / 2
            if i % 2 == 0:
                flux[mask] -= 0.05  # Deep (5%)
            else:
                flux[mask] -= 0.01  # Shallow (1%)

        flag, sigma, hard = test_odd_even(
            time, flux, transit_times, period, duration, sigma_threshold=3.0
        )

        assert flag  # Should detect the difference
        assert hard  # Should hard-reject


class TestPhysicalLimits:
    """Tests for astra.vetting.physical_limits."""

    def test_jupiter_size_passes(self):
        """Jupiter-sized planet should pass physical limits."""
        from astra.vetting.physical_limits import test_physical_limits

        flag, cause, rp, teq, hard = test_physical_limits(
            depth=0.01, period=3.0, r_star=1.0, teff=5778.0
        )

        assert not hard
        assert rp < 25.0  # ~11 R_earth

    def test_stellar_companion_fails(self):
        """Very deep transit implies stellar companion → hard reject."""
        from astra.vetting.physical_limits import test_physical_limits

        # depth=0.25 → Rp = sqrt(0.25) * 109 = 54.5 R_earth
        flag, cause, rp, teq, hard = test_physical_limits(
            depth=0.25, period=3.0, r_star=1.0, teff=5778.0
        )

        assert hard
        assert cause == "STELLAR_COMPANION"
        assert rp > 25.0
