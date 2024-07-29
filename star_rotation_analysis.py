"""
    Created on September 20 2021
    
    Description: This routine performs a Gaussian Process analysis of some activity indicator to constrain the rotation period of the star.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python star_rotation_analysis.py --gp_priors=data/priors.pars --outdir=./results/ --pairsplot=TOI-1759_blong_gp_pairsplot.png --input=data/TOI-1759_blong.rdb --nsteps=1000 --walkers=50 --burnin=200 -vpe
    
    python star_rotation_analysis.py --gp_priors=/Volumes/Samsung_T5/Science/TOI-1718/gp_phot_priors.pars --outdir=/Volumes/Samsung_T5/Science/TOI-1718/GP-phot/ --pairsplot=TOI-1718_tess_gp_pairsplot.png --input=/Volumes/Samsung_T5/Science/TOI-1718/TOI-1718_tessphot.rdb --nsteps=1000 --walkers=50 --burnin=200 -vpe
    
    python /Volumes/Samsung_T5/Science/star-activity-tools/star_rotation_analysis.py --gp_priors=gp_sindex_priors.pars --outdir=./ --pairsplot=TOI-1736_sindex_gp_pairsplot.png --input=TOI1736_sophie_sindex.txt --nsteps=1000 --walkers=50 --burnin=200 --variable_name="S-index" -vper
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys

from optparse import OptionParser
import numpy as np
from astropy.io import ascii

import priorslib
import gp_lib
import timeseries_lib

parser = OptionParser()
parser.add_option("-f", "--file_params", dest="file_params", help='Text file containing the default parameters the used wants to set. Any opion in the file will be bypassed by the option provided at runtime.',type='string',default="")
parser.add_option("-i", "--input", dest="input", help='Input time series data file (.rdb)',type='string',default="")
parser.add_option("-m", "--variable_name", dest="variable_name", help='Variable name',type='string',default="B$_l$")
parser.add_option("-u", "--variable_units", dest="variable_units", help='Variable units',type='string',default="")
parser.add_option("-o", "--outdir", dest="outdir", help="Directory to save output files",type='string',default="./")
parser.add_option("-g", "--gp_priors", dest="gp_priors", help='GP priors file',type='string',default="")
parser.add_option("-t", "--pairsplot", dest="pairsplot", help='Pairsplot file name',type='string',default="dummy")
parser.add_option("-n", "--nsteps", dest="nsteps", help="Number of MCMC steps",type='int',default=500)
parser.add_option("-w", "--walkers", dest="walkers", help="Number of MCMC walkers",type='int',default=32)
parser.add_option("-b", "--burnin", dest="burnin", help="Number of MCMC burn-in samples",type='int',default=100)
parser.add_option("-c", "--nbcores", dest="nbcores", help="Number of cores to run the MCMC in parallel",type='int',default=1)
parser.add_option("-r", action="store_true", dest="use_gls_period", help="Use period from GLS periodogram", default=False)
parser.add_option("-s", action="store_true", dest="savesamples", help="Save MCMC samples into file", default=False)
parser.add_option("-e", action="store_true", dest="mode", help="Best fit parameters obtained by the mode instead of median", default=False)
parser.add_option("-d", action="store_true", dest="plot_distributions", help="Plot detailed distributions", default=False)
parser.add_option("-l", action="store_true", dest="print_latex", help="print data in latex format",default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with star_rotation_analysis.py -h "); sys.exit(1);

## Load the default options if provided
## I update the default value of the parser, then reparse the input
if options.file_params!="":
    f = open(options.file_params, 'r')
    updated_args = []
    for line in f.readlines():
        if line.strip()=='': continue
        if line.strip()[0]=='#': continue
        sl = line.split()
        if sl[0] not in parser.defaults:
            print("Error: the default parameter file contains arguments not found in the options"); sys.exit(1);
        if sl[0] in updated_args:
            print("Error: argument repeated in default parameter file"); sys.exit(1);
        value = sl[1]
        if value.lower()=='true': value = True
        elif value.lower()=='false': value = False
        parser.defaults[sl[0]] = value
        updated_args.append(sl[0])
    f.close()
try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with star_rotation_analysis.py -h "); sys.exit(1);

if options.verbose:
    print('Input time series data file (.rdb): ', options.input)
    print('Variable name: ', options.variable_name)
    print('Variable units: ', options.variable_units)
    print('GP priors file: ', options.gp_priors)
    if options.pairsplot != "" :
        print('Pairsplot file name: ', options.pairsplot)
    print('Number of MCMC steps: ', options.nsteps)
    print('Number of MCMC walkers: ', options.walkers)
    print('Number of MCMC burn-in samples: ', options.burnin)
    print('Number of cores requested for MCMC: ', options.nbcores)

prior_path = os.path.abspath(options.gp_priors)
prior_basename = os.path.basename(prior_path)
gp_posterior = options.outdir + '/' + prior_basename.replace(".pars", "_posterior.pars")
output_pairsplot = options.outdir + '/' + options.pairsplot

if options.verbose:
    print("Output GP posterior: ", gp_posterior)

# Load gp parameters priors
gp_priors = priorslib.read_priors(options.gp_priors)
gp_params = priorslib.read_starrot_gp_params(gp_priors)
param_lim, param_fixed = {}, {}
# print out gp priors
print("----------------")
print("Input GP parameters:")
for key in gp_params.keys() :
    if ("_err" not in key) and ("_pdf" not in key) :
        param_fixed[key] = False
        pdf_key = "{0}_pdf".format(key)
        if gp_params[pdf_key] == "FIXED" :
            print("{0} = {1} ({2})".format(key, gp_params[key], gp_params[pdf_key]))
            param_lim[key] = (gp_params[key],gp_params[key])
            param_fixed[key] = True
        elif gp_params[pdf_key] == "Uniform" or gp_params[pdf_key] == "Jeffreys":
            error_key = "{0}_err".format(key)
            min = gp_params[error_key][0]
            max = gp_params[error_key][1]
            param_lim[key] = (min,max)
            print("{0} <= {1} ({2}) <= {3} ({4})".format(min, key, gp_params[key], max, gp_params[pdf_key]))
        elif gp_params[pdf_key] == "Normal" or gp_params[pdf_key] == "Normal_positive":
            error_key = "{0}_err".format(key)
            error = gp_params[error_key][1]
            param_lim[key] = (gp_params[key]-5*error,gp_params[key]+5*error)
            print("{0} = {1} +/- {2} ({3})".format(key, gp_params[key], error, gp_params[pdf_key]))
print("----------------")


# PIC: Updating the approach to store the information of each parameter in a UNIQUE dictionary
gp_priors = priorslib.read_priors(options.gp_priors)
gp_params = priorslib.read_starrot_gp_params(gp_priors)
## All we need really is to have some "default" limits for the kernel the way it's implemented later
## So for now, let's just add limits to the dictionary:
## Get the name of the parameters in the dictionary:
params_list = []
for key in gp_params.keys() :
    if ("_err" not in key) and ("_pdf" not in key) :
        params_list.append(key)
## They for each of key, let's define limits
for key in params_list:
    ## Ok nope, there are three cases to implement
    if gp_params[key+'_pdf'] == "FIXED" :
        gp_params[key+'_lim'] = (gp_params[key],gp_params[key])
    elif gp_params[key+'_pdf'] == "Uniform" or gp_params[key+'_pdf'] == "Jeffreys":
        gp_params[key+'_lim'] = (gp_params[key+'_err'][0],gp_params[key+'_err'][1])
    elif gp_params[key+'_pdf'] == "Normal" or gp_params[key+'_pdf'] == "Normal_positive":
        gp_params[key+'_lim'] = (gp_params[key]-5*gp_params[key+'_err'][1],gp_params[key]+5*gp_params[key+'_err'][1])
##########################################
### LOAD input data
##########################################
if options.verbose:
    print("Loading time series data ...")
timeseriesdata = ascii.read(options.input)
bjds, y, yerr = timeseriesdata["col1"], timeseriesdata["col2"], timeseriesdata["col3"]

if options.print_latex :
    # to print the blong data in latext format:
    for i in range(len(y)) :
        print("{0:.7f} & ${1:.1f}\pm{2:.1f}$ \\\\".format(bjds[i],y[i],yerr[i]))

t0 = np.nanmin(bjds)
if options.verbose :
    print("T0 = {:.6f} BJD".format(t0))

##########################################
### ANALYSIS of input data
##########################################
ylabel = r"{}".format(options.variable_name)
if options.variable_units != "" :
    ylabel = r"{0} [{1}]".format(options.variable_name,options.variable_units)
    
best_period = gp_params["period"]
    
gls = timeseries_lib.periodogram(bjds, y, yerr, nyquist_factor=4, probabilities = [0.00001], npeaks=1, y_label=ylabel, plot=options.plot, phaseplot=options.plot)

if options.use_gls_period :
    best_period = gls['period']

    if options.verbose :
        print("GLS highest peak at P={:.3f} d".format(best_period))
        
else :
    if options.verbose :
        print("Initial guess period at P={:.3f} d".format(best_period))

amplitude = gp_params["amplitude"]
decaytime = gp_params["decaytime"]
smoothfactor = gp_params["smoothfactor"]

fit_mean = True
if param_fixed["mean"] :
    fit_mean = False

fit_white_noise = True
if param_fixed["white_noise"] :
    fit_white_noise = False

fix_period = param_fixed["period"]
period_lim = param_lim["period"]

fix_amplitude = param_fixed["amplitude"]
amplitude_lim = param_lim["amplitude"]

fix_decaytime = param_fixed["decaytime"]
decaytime_lim = param_lim["decaytime"]

fix_smoothfactor = param_fixed["smoothfactor"]
smoothfactor_lim = param_lim["smoothfactor"]

amp, nwalkers, niter, burnin = 1e-4, options.walkers, options.nsteps, options.burnin
nbcores = options.nbcores

# Run GP on B-long data with a QP kernel
gp = gp_lib.star_rotation_gp(bjds, y, yerr, period=best_period, period_lim=period_lim, 
                             fix_period=fix_period, amplitude=amplitude, amplitude_lim=amplitude_lim, 
                             fix_amplitude=fix_amplitude, decaytime=decaytime, decaytime_lim=decaytime_lim, 
                             fix_decaytime=fix_decaytime, smoothfactor=smoothfactor, 
                             smoothfactor_lim=smoothfactor_lim, fix_smoothfactor=fix_smoothfactor, 
                             fixpars_before_fit=True, fit_mean=fit_mean, fit_white_noise=fit_white_noise, 
                             period_label=r"Prot [d]", amplitude_label=r"$\alpha$ {0}".format(options.variable_units), 
                             decaytime_label=r"$l$ [d]", smoothfactor_label=r"$\beta$", 
                             mean_label=r"$\mu$ {0}".format(options.variable_units), 
                             white_noise_label=r"$\sigma$ {0}".format(options.variable_units), 
                             output_pairsplot=output_pairsplot, run_mcmc=True, amp=amp, nwalkers=nwalkers, 
                             niter=niter, burnin=burnin, x_label="BJD", y_label=ylabel, 
                             output=gp_posterior, best_fit_from_mode = options.mode, 
                             plot_distributions = options.plot_distributions, 
                             plot=True, verbose=True, gp_params=gp_params, params_list=params_list, nbcores=nbcores)

gp_params = gp_lib.get_star_rotation_gp_params(gp)
best_period = gp_params["period"]

timeseries_lib.phase_plot(bjds, y, yerr, gp, best_period, ylabel=ylabel, t0=t0, alpha=1)
