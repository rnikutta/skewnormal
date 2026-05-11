"""Worked example of the skewnormal package.

Estimate the skew-normal parameters from a (median, dL, dR, fraction)
input, draw a sample from the fitted PDF, and plot the CDF, PDF and a
normalised histogram of the sample.

Run from the command line:

    $ python3 example.py

or import and call interactively:

    >>> from example import plot_skewnorm
    >>> mu, sigma, alpha, sample = plot_skewnorm()
"""

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "2026.05"

import numpy as np
import matplotlib.pyplot as plt
import skewnormal


def plot_skewnorm(mid=2.0, dl=0.7, dr=1.2,
                  samplesize=100_000,
                  pdffile='skewnormal.pdf',
                  pngfile='skewnormal.png'):
    """Fit a skew-normal to ``mid +dr/-dl`` at the 90% interval, plot,
    save figures, and return the fit + sample."""

    # Find parameters of the skew-normal matching the input statistics
    mu, sigma, alpha = skewnormal.get_skewnormal_parameters(
        mid, dl, dr, fraction=0.9)

    # SkewNormal object: provides .pdf, .cdf, .rvs
    sn = skewnormal.SkewNormal(mu, sigma, alpha)
    sample = sn.rvs(size=int(samplesize))

    x = np.linspace(0, 5, 200)
    pdf = sn.pdf(x)
    cdf = sn.cdf(x)

    # Bounds of the 90% interval
    xl, xr = mid - dl, mid + dr

    # Convert fraction to known locations on the CDF
    prob_frac = 0.9
    Phixl = skewnormal.get_tail(prob_frac, mode='fraction')
    Phixr = 1.0 - Phixl

    # Figure styling
    print("Plotting...")
    fontsize = 11
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'figure.figsize':    (5, 3.5),
        'axes.labelsize':    fontsize,
        'axes.linewidth':    1.0,
        'axes.titlesize':    fontsize,
        'font.size':         fontsize,
        'legend.fontsize':   fontsize - 1,
        'xtick.labelsize':   fontsize,
        'ytick.labelsize':   fontsize,
        'xtick.major.size':  3,
        'xtick.minor.size':  1.5,
        'ytick.major.size':  3,
        'ytick.minor.size':  1.5,
        # don't use Type-3 fonts (e.g. MNRAS doesn't allow them)
        'ps.useafm':         True,
        'pdf.use14corefonts': True,
    })

    fig, ax = plt.subplots()

    # Grid of x,y reference lines on the CDF
    ls, lw, color = ':', 1, '0.2'
    for xv in (xl, mid, xr):
        ax.axvline(xv, ls=ls, lw=lw, color=color)
    for yv in (Phixl, 0.5, Phixr):
        ax.axhline(yv, ls=ls, lw=lw, color=color)

    # The three known points on the CDF
    ax.plot((xl, mid, xr), (Phixl, 0.5, Phixr),
            ls='none', marker='o', mec='k', mfc='none', mew=1.5, ms=7,
            label='known points on CDF', zorder=1)

    # PDF and CDF curves
    ax.plot(x, cdf, ls='-', color='b', lw=2,
            label=r'fitted CDF $\Phi(x|\mu,\sigma,\alpha)$', zorder=0)
    ax.plot(x, pdf, 'r-', lw=2,
            label=r'its PDF $\phi(x|\mu,\sigma,\alpha)$', zorder=0)

    # Histogram of drawn sample (matplotlib 3.x: density= replaces normed=)
    ax.hist(sample, bins=50, histtype='stepfilled',
            color='0.2', ec='k', alpha=0.15, density=True,
            label=fr'${int(samplesize):,}$ sample from $\phi(x)$')

    leg = ax.legend(loc='center right', frameon=True)
    leg.get_frame().set_facecolor('w')
    leg.get_frame().set_edgecolor('w')
    leg.get_frame().set_alpha(0.7)

    ax.set_xlim(mid - 3 * dl, mid + 3 * dr)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('PDF, CDF')

    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.97)

    if pdffile:
        print(f"Saving PDF figure to {pdffile}")
        plt.savefig(pdffile)
    if pngfile:
        print(f"Saving PNG figure to {pngfile}")
        plt.savefig(pngfile, dpi=200)

    return mu, sigma, alpha, sample


if __name__ == "__main__":
    mu, sigma, alpha, sample = plot_skewnorm()
    print(f"Fitted: mu={mu:.4f}, sigma={sigma:.4f}, alpha={alpha:.4f}")
    print(f"Sample size: {len(sample)}; median={np.median(sample):.3f}")
