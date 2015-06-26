"""This is part if the skewnormal software.

To run this example:

$ipython
 In[1]: import example, skewnormal
 In[2]: mu, sig, alpha, sample, dist = example.plot_skewnorm()  # estimated free params, 1e5 sample, sampling object
 In[3]: sample = dist.random() # draw a new sample of 1e5 variates from the estimated skew-normal PDF

"""

__author__ = "Robert Nikutta <robert.nikutta@gmail.com"
__version__ = "2015-06-26"

import numpy as N
import pylab as p
from matplotlib.ticker import MaxNLocator
import skewnormal

# show how it all works
def plot_skewnorm(mid=2.,dl=0.7,dr=1.2,samplesize=1e5,pdffile='skewnormal.pdf'):

    # find params of our skew normal; get a sample for histogram
    mu,sig,alpha,dist = skewnormal.get_skewnormal_parameters(mid,dl,dr,fraction=0.9,returndist=True,size=samplesize)
    sample = dist.random()

    # PyMC SkewNormal; for plotting PDF and CDF
    sn = skewnormal.SkewNormal(mu,sig,alpha)
    x = N.linspace(0,5,100)
    pdf = sn.pdf(x)
    cdf = sn.cdf(x)

    # left and right ranges
    xl = mid - dl
    xr = mid + dr

    # convert fraction to known locations of the CDF
    prob_frac = 0.9
    Phixl = skewnormal.get_tail(prob_frac,mode='fraction')
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
    
    return mu, sig, alpha, sample, dist
