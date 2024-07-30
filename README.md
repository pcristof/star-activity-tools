# star-activity-tools

## Updates

P. I. Cristofari, Jul. 2024
Updated version built from a fork of the original repository.
Fixed several things, attempt to make things more transparent and reliable.

Summary of modifications will appear here:
- User priors are now used (no update of the prior strategy based on kernel within the code).
- Added support for Apple M chips with parallel emcee.
- added options such as `-f`, reading an input file storing all the details of the requested computation.



## Original README.md

Toolkit for analyzing stellar activity indicators.
 
The main routine `star_rotation_analysis.py` performs a Quasi-Periodic Gaussian Process analysis of some activity indicator to constrain the star's rotation period.

Below is an example of simple use to run this tool to analyze the time series of the longitudinal magnetic field measured with SPIRou. In this example the data is saved in the file `data/TOI-1759_blong.rdb` and the priors are saved in the file `data/priors.pars`:

```
python star_rotation_analysis.py --gp_priors=data/priors.pars --outdir=./results/ 
--pairsplot=TOI-1759_blong_gp_pairsplot.png --input=data/TOI-1759_blong.rdb 
--nsteps=1000 --walkers=50 --burnin=200 -vpe
```

If you're using this code, please cite [Martioli et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220201259M/abstract)
