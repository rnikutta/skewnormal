"""skewnormal — Skew-Normal distribution: parameter estimation and sampling.

Given the median and (possibly asymmetric) lower/upper spreads of a sample
known to come from a Skew-Normal distribution, recover the three free
parameters (mu, sigma, alpha) of the underlying PDF.

Original implementation: Robert Nikutta, 2015 (Python 2; PyMC-based sampling).
Python-3 modernization, 2026.

License: BSD 3-clause.
"""

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "2026.05"

import numpy as np
from scipy.stats import skewnorm as _skewnorm
from scipy.optimize import minimize
from scipy.special import erf, owens_t


# ---------------------------------------------------------------------------
# Skew-Normal distribution: thin wrapper around scipy.stats.skewnorm
# ---------------------------------------------------------------------------

class SkewNormal:
    """Skew-Normal PDF/CDF/sampler with parameters ``(mu, sigma, alpha)``.

    The skew-normal distribution is parameterised here by its location
    ``mu``, scale ``sigma > 0``, and shape (skewness) ``alpha`` (the
    Azzalini parameterisation).  ``alpha = 0`` recovers a standard normal.
    """

    def __init__(self, mu, sigma, alpha):
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0; got {sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)

    def __repr__(self):
        return (f"SkewNormal(mu={self.mu:.4g}, sigma={self.sigma:.4g}, "
                f"alpha={self.alpha:.4g})")

    def pdf(self, x):
        """Probability density at ``x``."""
        return _skewnorm.pdf(x, self.alpha, loc=self.mu, scale=self.sigma)

    def cdf(self, x):
        """Cumulative distribution at ``x``."""
        return _skewnorm.cdf(x, self.alpha, loc=self.mu, scale=self.sigma)

    def rvs(self, size=1, random_state=None):
        """Draw ``size`` random variates."""
        return _skewnorm.rvs(self.alpha, loc=self.mu, scale=self.sigma,
                             size=size, random_state=random_state)


# ---------------------------------------------------------------------------
# Probability/sigma <-> tail conversions
# ---------------------------------------------------------------------------

def get_tail(arg, mode='sigma'):
    """Convert between sigma-multiples (or probability fractions) and tail prob.

    Parameters
    ----------
    arg : float
        If ``mode='sigma'``, multiple of sigma (e.g. ``arg=1`` => 1 sigma).
        If ``mode='fraction'``, fraction of central probability (e.g.
        ``arg=0.683`` for 1 sigma, ``arg=0.9`` for 90 %).
    mode : {'sigma', 'fraction'}

    Returns
    -------
    tail : float
        One-sided tail probability.
    """
    if mode == 'sigma':
        return 0.5 * (1.0 - erf(arg / np.sqrt(2.0)))
    if mode == 'fraction':
        return 0.5 * (1.0 - arg)
    raise ValueError(f"mode must be 'sigma' or 'fraction'; got {mode!r}")


# ---------------------------------------------------------------------------
# Parameter estimator: solve (mu, sigma, alpha) from (xM, dL, dR, fraction)
# ---------------------------------------------------------------------------

def get_skewnormal_parameters(xM, dL, dR, fraction=0.683, verbose=False):
    """Estimate ``(mu, sigma, alpha)`` of a Skew-Normal PDF from sample stats.

    The CDF of the fitted Skew-Normal is required to pass through three
    known points:

        (xM - dL, (1 - fraction) / 2)
        (xM,      0.5)
        (xM + dR, (1 + fraction) / 2)

    Parameters
    ----------
    xM : float
        Sample median.
    dL : float
        Lower spread (median minus lower-bound x-value). Must be > 0.
    dR : float
        Upper spread (upper-bound x-value minus median). Must be > 0.
    fraction : float, default 0.683
        Fraction of total probability covered by ``[xM-dL, xM+dR]``.
        Common values: 0.683 (1-sigma Gaussian), 0.9 (90 %, chi-sq dchi=2.71),
        0.95 (95 %).
    verbose : bool, default False
        Pass ``True`` to print solver progress.

    Returns
    -------
    mu, sigma, alpha : float
        Skew-Normal location, scale, and shape parameters.

    Notes
    -----
    For sigma+/sigma- ratios above ~5-8 the Skew-Normal family is not
    flexible enough to satisfy all three CDF constraints exactly.  The
    solver returns its best-fit; callers can detect this by checking
    the CDF residuals at the three input target points.

    Examples
    --------
    >>> mu, sigma, alpha = get_skewnormal_parameters(2.1, 0.8, 1.3, fraction=0.9)
    """
    if dL <= 0 or dR <= 0:
        raise ValueError(f"dL ({dL}) and dR ({dR}) must be > 0")
    if not 0 < fraction < 1:
        raise ValueError(f"fraction ({fraction}) must be in (0, 1)")

    xL, xR = xM - dL, xM + dR
    tail = get_tail(fraction, mode='fraction')
    x_targets = np.array([xL, xM, xR])
    p_targets = np.array([tail, 0.5, 1.0 - tail])

    # Initial guess: skew alpha proportional to the asymmetry of the spreads
    alpha0 = np.sign(dR - dL) * (max(dL, dR) / min(dL, dR))
    sigma0 = 0.5 * (dL + dR)
    init = (xM, sigma0, alpha0)

    def loss(params):
        mu, sigma, alpha = params
        if sigma <= 0:
            return 1e10
        cdf = _skewnorm.cdf(x_targets, alpha, loc=mu, scale=sigma)
        return float(np.sum((cdf - p_targets) ** 2))

    res = minimize(loss, init, method='Nelder-Mead',
                   options={'xatol': 1e-8, 'fatol': 1e-12,
                            'maxiter': 5000, 'disp': verbose})
    mu, sigma, alpha = res.x
    return mu, sigma, alpha


# ---------------------------------------------------------------------------
# Backward-compatibility shims
# ---------------------------------------------------------------------------

def pdfSN(x, mu, sig, alpha):
    """Skew-Normal PDF.  Prefer ``scipy.stats.skewnorm.pdf`` or
    :class:`SkewNormal` for new code."""
    return _skewnorm.pdf(x, alpha, loc=mu, scale=sig)


def cdfSN(x, mu, sig, alpha):
    """Skew-Normal CDF.  Prefer ``scipy.stats.skewnorm.cdf`` or
    :class:`SkewNormal` for new code."""
    return _skewnorm.cdf(x, alpha, loc=mu, scale=sig)


def OwenT(h, a):
    """Owen's T-function.  Thin wrapper over :func:`scipy.special.owens_t`,
    bundled with SciPy since v1.4."""
    return owens_t(h, a)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== skewnormal v{__version__} self-test ===\n")

    # Reproduce the example from the original example.py
    xM, dL, dR, frac = 2.1, 0.8, 1.3, 0.9
    print(f"Input: xM={xM}, dL={dL}, dR={dR}, fraction={frac}")
    mu, sigma, alpha = get_skewnormal_parameters(xM, dL, dR, fraction=frac)
    print(f"  Solved: mu={mu:.4f}, sigma={sigma:.4f}, alpha={alpha:.4f}")

    sn = SkewNormal(mu, sigma, alpha)
    np.random.seed(42)
    sample = sn.rvs(size=100_000)
    lo, med, hi = np.percentile(sample,
                                [(1-frac)/2*100, 50, (1+frac)/2*100])
    print(f"  Sample percentiles ({frac*100:.0f}% interval): "
          f"[{lo:.3f}, {med:.3f}, {hi:.3f}]")
    print(f"  Expected: [{xM-dL:.3f}, {xM:.3f}, {xM+dR:.3f}]")

    err = max(abs(lo - (xM-dL)), abs(med - xM), abs(hi - (xM+dR)))
    print(f"  Max abs error: {err:.4f}")
    print(f"  Status: {'PASS' if err < 0.01 else 'FAIL'}")
