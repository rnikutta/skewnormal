skewnormal
==========

**Author:** Robert Nikutta

**Version:** 2014-08-19

**License:** BSD 3-clause, please see [LICENSE](./LICENSE) file

<img src="./skewnormal.png" alt="image" style="width:400">

Estimate the free parameters (mu,sigma,alpha) of a Skew-Normal PDF
when only the location and (possibly asymmetric) spread of a sample
drawn from that PDF are known. Also provides a PyMC sampling object to
draw random variates from the estimated Skew-Normal PDF. Includes all
required helper functions, e.g. Owen's T function. See docstring of
classes and funcs for usage.

Run `plot_skewnorm()` to see an example of how it all works (it
generates a PDF figure, a PNG version of which you can see above).
