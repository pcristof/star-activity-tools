# star-activity-tools

Tool kit for the analysis of stellar activity indicators. 

The main routine `star_rotation_analysis.py` performs a Quasi-Periodic Gaussian Process analysis of some activity indicator to constrain the rotation period of the star. 

Below is a simple usage example to run this tool to anlyze the longitudinal magnetic field time-series measured with SPIRou. The data is saved in a file `data/TOI-1759_blong.rdb` and the priors are saved in the file `data/priors.pars`:

```
python star_rotation_analysis.py --gp_priors=data/priors.pars --outdir=./results/ --pairsplot=TOI-1759_blong_gp_pairsplot.png --input=data/TOI-1759_blong.rdb --nsteps=1000 --walkers=50 --burnin=200 -vp

```
