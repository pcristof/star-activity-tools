# star-activity-tools

Toolkit for analyzing stellar activity indicators.
 
The main routine `star_rotation_analysis.py` performs a Quasi-Periodic Gaussian Process analysis of some activity indicator to constrain the star's rotation period.

Below is an example of simple use to run this tool to analyze the time series of the longitudinal magnetic field measured with SPIRou. In this example the data is saved in the file `data/TOI-1759_blong.rdb` and the priors are saved in the file `data/priors.pars`:

```
python star_rotation_analysis.py --gp_priors=data/priors.pars --outdir=./results/ 
--pairsplot=TOI-1759_blong_gp_pairsplot.png --input=data/TOI-1759_blong.rdb 
--nsteps=1000 --walkers=50 --burnin=200 -vpe
```

If you're using this code, please cite [Martioli et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220201259M/abstract)
