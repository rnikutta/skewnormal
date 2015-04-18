"""Skew-Normal class and helper functions.

Allows to estimate the 3 free parameters of a Skew-Normal PDF when
knowing only its measured median and (asymmetric) spread.

Includes all required helper functions, e.g. Owen's T function.

See docstring of classes and funcs for usage. See example.py file and
run plot_skewnorm() there to see an example of how it all works (it
generates a PDF figure).

"""

__author__ = "Robert Nikutta <robert.nikutta@gmail.com"
__version__ = "2015-04-18"

import numpy as N
from scipy.special import erf, erfc
from scipy.optimize import fmin
import pymc


class SkewNormal:

    """Class for Skew-Normal distribution. Provides PDF, CDF.

    Initialize with three parameters mu, sig, alpha.

    Arguments:
    ----------
    mu, sig, alpha : float tuple
        The three free parameters of a Skew-Normal PDF (mean,spread,skewness).

    """

    def __init__(self,mu,sig,alpha):
        self.mu = mu
        self.sig = sig
        self.alpha = alpha

    def pdf(self,x):
        """PDF of Skew-Normal distribution"""

        return pdfSN(x,self.mu,self.sig,self.alpha)

    def cdf(self,x):
        """CDF of Skew-Normal distribution"""

        return cdfSN(x,self.mu,self.sig,self.alpha)


def pdfSN(x,mu,sig,alpha):
    """PDF of Skew-Normal distribution"""

    aux1 = 1. / (sig*N.sqrt(2*N.pi))                 # normalization
    aux2 = N.exp( -(x-mu)**2 / (2*sig**2) )          # Gaussian
    aux3 = erfc( -alpha*(x-mu) / (sig*N.sqrt(2)) )   # Erfc

    return aux1 * aux2 * aux3


def cdfSN(x,mu,sig,alpha):
    """CDF of Skew-Normal distribution"""

    h = (x-mu) / sig
    return 0.5 * erfc(-(x-mu)/(sig*N.sqrt(2))) - 2*OwenT(h,alpha)
    

def OwenT(h,alpha):
    """Owen's T-function.

    Compute Owen's T-function for given value(s) of h and a given value of alpha:

                      1                exp[ -0.5*(h^2) * (1+x^2) ]
        T(h,alpha) = ---- \int_0^alpha --------------------------- dx
                     2 pi                        1 + x^2

    Owen's T-function is used, e.g., in the CDF of the Skew Normal distribution.

    For definition see, e.g.:
        http://en.wikipedia.org/wiki/Owen%27s_T_function
        http://reference.wolfram.com/language/ref/OwenT.html

    Parameters:
    -----------
    h : float or 1-dimensional sequence of floats (list,tuple,ndarray)
        h parameter (see definition). If it is a sequence, T(h,alpha)
        will return a sequence of results.

    alpha : float
        Integration in Owen's T-function runs from 0 to alpha.

    Returns:
    --------
    T(h,alpha) for all value(s) of h.

    """

    from scipy.integrate import romberg

    if not isinstance(h,(list,tuple,N.ndarray)):
        try:
            h = [h]
        except:
            raise Exception

    if not isinstance(h,N.ndarray):
        try:
            h = array(h)
        except:
            raise Exception

    def _integrand(x,h):
        aux = 1. + x**2
        return N.exp(-0.5 * h**2 * aux) / aux

    result = N.zeros(h.size)
    for j in xrange(result.size):
        result[j] = romberg(_integrand,0.,alpha,args=(h[j],)) / (2*N.pi)        

    return result


def get_tail(arg,mode='sigma'):
    """Remainder of the half-interval corresponding to 'arg' sigma of a PDF.

    Arguments:
    ----------
    arg : float
        Either "how many sigmas" (if mode='sigma'), or "the fraction of
        the total probability under the normalized PDF (if
        mode='fraction').

    mode : str
       Either 'sigma' or 'fraction'. See description of arg.

    Returns:
    --------
    tail : float
    """
    

    if mode == 'sigma':   # how many sigmas away from median?
        fraction = erf(arg/N.sqrt(2))
    elif mode == 'fraction':   # fraction of total probability under the normalized PDF?
        fraction = arg
    else:
        raise Exception, "mode must be either 'sigma' or 'fraction'."

    tail = (1.0-fraction)/2.  # half-interval

    return tail


def get_skewnormal_parameters(xM,dL,dR,fraction=0.9,verbose=True,returndist=False):

    """Estimate the free parameters mu,sigma,alpha of a Skew-Normal PDF from measured sample statistics.

    Arguments:
    ----------
    The input is xM+dR/-dL, where

    xM : float
        Measured median of a sample known to be from drawn from an
        unknown Skew Normal PDF.

    dL : float
        Left-sided spread of the sample. The 

    dR : float
        Right-sided spread of the sample.

    fraction : float (in [0,1])
        The fraction of the normalized total probability under the PDF
        that the interval [xM-dL,xM+dR] corresponds to. Many data
        analysis tools report for instances a "1-sigma interval around
        the median" (fraction=0.683 for Normal), or a "90 percent"
        interval (corresponding to an increase of exactly +1 around
        the minimum for a chi-squared statistic), etc.

    verbose: bool
        If True, print more verbose messages.

    returndist : bool (default False)
        If True, also return a PyMC sampling object. See docstring of
        get_sample() for sampling usage.

    Returns:
    --------
    mu, sigma, alpha : float tuple
        The three free estimated parameters of Skew-Normal PDF (mean,
        spread, skewness).

    dist : obj
        A PyMC sampling object (if returndist=True). Allows to draw a
        random sample with statistics conforming to the estimated PDF.

    Example:
    --------
    xM, dL, dR, fraction = 2.1, 0.8, 1.3, 0.9
    mu, sigma, alpha, dist = get_skewnormal_parameters(xM,dL,dR,fraction,returndist=True)

    See get_sample() docstring for generating a sample from that the
    estimated PDF.
    """

    assert (N.array((dL,dR,fraction)) > 0.).all(), "dL, dR, fraction, samplesize must all be > 0"

    # input range of parameter x, corresponding to a PDF fraction of sigsize
    xL = xM - dL   # loc of left bound
    xR = xM + dR   # loc of right bound

    # get x and y (vectors) of known data points in the CDF; they will be fitted
    tail = get_tail(fraction,mode='fraction')
    xd = N.array((xL,xM,xR))          # data: left, median, right
    yd = N.array((tail,0.5,1-tail))   # the CDF at those x-values

    # find parameters mu, sigma, alpha of a matching SkewNormal distribution via minimization
    if verbose:
        print "Finding optimal parameters mu, sig, alpha of SkewNormal distribution..."

    # educated initial guess for mu, sigma, alpha parameters of the SkewNormal distribution
    init = (xM, 0.5*(abs(xL)+abs(xR)), N.sign(dR-dL)*(max(dL,dR)/min(dL,dR)))

    def minimizeme(x,*args):
        mu, sig, alpha = x
        CDF = cdfSN(xd,mu,sig,alpha)
        return ((CDF-yd)**2).sum()  # sum of least squares

    mu_,sig_,alpha_ = fmin(minimizeme,init,args=(xd,yd),xtol=1.e-6,disp=verbose)

    if returndist:
        sn = pymc.SkewNormal('sn',mu_,1/sig_**2,alpha_)
        return mu_, sig_, alpha_, sn
    else:
        return mu_, sig_, alpha_


def get_sample(distribution,samplesize,verbose=True):

    # sanity checks for input
    try:
        samplesize = int(samplesize)
    except:
        raise Exception, "'samplesize' can not be converted to integer."

    assert (samplesize > 0.), "samplesize must all be > 0"

    # Draw samples via PyMC
    # Not using the 'proper' PyMC way, i.e. to generate a model, and
    # sample from it via MCMC, because this is twice a slow in our
    # case, where we only want to sample from a single variate.
    if verbose:
        print "Drawing %d samples from a SkewNormal distribution..." % samplesize

    sample = N.array([distribution.rand() for j in range(samplesize)])

    return sample


# show how it all works
def plot_skewnorm(mid=2.,dl=0.7,dr=1.2,samplesize=1e5,pdffile='skewnormal.pdf'):

    import pylab as p
    from matplotlib.ticker import MaxNLocator

    # find params of our skew normal; get a sample for histogram
    mu,sig,alpha,dist = get_skewnormal_parameters(mid,dl,dr,fraction=0.9,returndist=True)
    sample = get_sample(dist,samplesize)

    # PyMC SkewNormal; for plotting PDF and CDF
    sn = SkewNormal(mu,sig,alpha)
    x = N.linspace(0,5,100)
    pdf = sn.pdf(x)
    cdf = sn.cdf(x)

    # left and right ranges
    xl = mid - dl
    xr = mid + dr

    # convert fraction to known locations of the CDF
    prob_frac = 0.9
    Phixl = get_tail(prob_frac,mode='fraction')
    Phixr = 1-Phixl

    # Set up figure
    print "Plotting..."
    fontsize = 11
    fig_size = (5,3.5)
    p.rcParams['font.family'] = 'sans-serif'
    p.rcParams['figure.figsize'] = fig_size # #fig_size
    p.rcParams['axes.labelsize'] = fontsize
    p.rcParams['axes.linewidth'] = 1.
    p.rcParams['axes.titlesize'] = fontsize
    p.rcParams['font.size'] = fontsize
    p.rcParams['legend.fontsize'] = fontsize-1
    p.rcParams['xtick.labelsize'] = fontsize
    p.rcParams['ytick.labelsize'] = fontsize
    p.rcParams['xtick.major.size'] = 3      # major tick size in points
    p.rcParams['xtick.minor.size'] = 1.5      # major tick size in points
    p.rcParams['ytick.major.size'] = 3      # major tick size in points
    p.rcParams['ytick.minor.size'] = 1.5      # major tick size in points

    # don't use Type 3 fonts (e.g. MNRAS doesn't allow them)
    p.rcParams['ps.useafm'] = True
    p.rcParams['pdf.use14corefonts'] = True
    p.rcParams['text.usetex'] = True
    p.rcParams['font.family'] = 'sans-serif'

    # Plot
    fig = p.figure()
    ax = fig.add_subplot(111)

    # grid of x.y points on the CDF
    ls, lw, color = ':', 1, '0.2'
    ax.axvline(xl,ls=ls,lw=lw,color=color)
    ax.axvline(xr,ls=ls,lw=lw,color=color)
    ax.axhline(Phixl,ls=ls,lw=lw,color=color)
    ax.axhline(Phixr,ls=ls,lw=lw,color=color)
    ax.axvline(mid,ls=ls,lw=lw,color=color)
    ax.axhline(0.5,ls=ls,lw=lw,color=color)

    # the three known points on the CDF
    ax.plot((xl,mid,xr),(Phixl,0.5,Phixr),ls='none',marker='o',mec='k',mfc='none',mew=1.5,ms=7,label=r'known points on CDF',zorder=1)

    # PDF and CDF
    ax.plot(x,cdf,ls='-',color='b',lw=2,label=r'fitted CDF \Phi(x|\mu,\sigma,\alpha)$',zorder=0)
    ax.plot(x,pdf,'r-',lw=2,label=r'its PDF $\phi(x|\mu,\sigma,\alpha)$',zorder=0)

    # histogram of drawn sample
    ax.hist(sample,bins=50,histtype='stepfilled',color='0.2',ec='k',alpha=0.15,normed=True,label=r'$10^5$ sample from $\phi(x)$')

    # legend
    leg = p.legend(loc='center right',frameon=1)
    leg.get_frame().set_facecolor('w')
    leg.get_frame().set_edgecolor('w')  #A6DAFF
    leg.get_frame().set_alpha(0.7)

    # limits, labels
    ax.set_xlim(mid-3*dl,mid+3*dr)
    ax.set_ylim(0,1)
    ax.set_xlabel('x')
    ax.set_ylabel('PDF, CDF')

    # margin adjustments
    p.subplots_adjust(left=0.1,right=0.98,bottom=0.12,top=0.97)

    # save figure
    print "Saving figure to PDF file", pdffile
    p.savefig(pdffile)
#    p.savefig("skewnormal.png",dpi=200)
