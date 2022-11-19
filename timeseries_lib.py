# -*- coding: utf-8 -*-
"""
    Created on Sep 21 2021
    
    Description: library with utilities for the analysis of time series
    
    @authors:  Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.timeseries import LombScargle
from PyAstronomy.pyasl import foldAt


def periodogram(x, y, yerr, period=0., nyquist_factor=20, probabilities = [0.01, 0.001], y_label="y", check_period=0, npeaks=1, phaseplot=False, plot=False, plot_frequencies=False) :
    """
        Description: calculate GLS periodogram
        """
    
    ls = LombScargle(x, y, yerr)

    frequency, power = ls.autopower(nyquist_factor=nyquist_factor)

    fap = ls.false_alarm_level(probabilities)
    
    if period == 0 :
        sorted = np.argsort(power)
        #best_frequency = frequency[np.argmax(power)]
        best_frequencies = frequency[sorted][-npeaks:]
        best_powers = power[sorted][-npeaks:]
        period = 1./best_frequencies
    else :
        best_frequencies = [1./period]
        best_powers = [np.nanmax(power)]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    if npeaks > 10 :
        print("ERROR: npeaks must be up to 10, exiting ... ")
        exit()

    periods = 1/frequency

    if plot :
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels

        if plot_frequencies :
            plt.plot(frequency, power, color="k", zorder=1)
        else :
            plt.plot(periods, power, color="k", zorder=1)
            
        for i in range(len(best_frequencies)) :
            best_frequency, best_power = best_frequencies[i], best_powers[i]
            if plot_frequencies :
                plt.vlines(best_frequency, np.min(power), best_power, ls="--", color=colors[i], label="Max power at F={0:.1f} 1/d".format(best_frequency), zorder=2)
            else :
                plt.vlines(1/best_frequency, np.min(power), best_power, ls="--", color=colors[i], label="Max power at P={0:.1f} d".format(1/best_frequency), zorder=2)

            #plt.hlines(best_power, np.min(periods), 1/best_frequency, ls="--", color=colors[i], zorder=2)
 
        if plot_frequencies :
            plt.xlabel("Frequency [1/d]", fontsize=15)
        else :
            if check_period :
                plt.vlines(check_period, np.min(power), np.max(power), color="red",ls="--", label="Max power at P={0:.4f} d".format(check_period))

            for i in range(len(fap)) :
                plt.text(np.min(periods),fap[i]+0.01,r"FAP={0:.3f}%".format(100*probabilities[i]),horizontalalignment='left', fontsize=15)
                plt.hlines([fap[i]], np.min(periods), np.max(periods),ls=":", lw=0.5)
                plt.xlabel("Period [d]", fontsize=15)
            
        #plt.yscale('log')
        plt.xscale('log')
        plt.ylabel("Power", fontsize=15)
        plt.legend(fontsize=16)
        plt.show()
    else :
        for i in range(len(best_frequencies)) :
            best_frequency, best_power = best_frequencies[i], best_powers[i]

    phases = foldAt(x, 1/best_frequency, T0=x[0])
    sortIndi = np.argsort(phases)

    if plot and phaseplot:
        plt.errorbar(phases[sortIndi],y[sortIndi],yerr=yerr[sortIndi],fmt='o', color="k")
        plt.ylabel(r"{}".format(y_label), fontsize=16)
        plt.xlabel("phase (P={0:.3f} d)".format(1/best_frequency), fontsize=16)
        plt.show()

    loc = {}
    if npeaks == 1 :
        loc['best_frequency'] = best_frequency
        loc['period'] = 1 / best_frequency
    else :
        loc['best_frequency'] = best_frequencies
        loc['period'] = 1 / best_frequencies
    loc['power'] = power
    loc['frequency'] = frequency
    loc['phases'] = phases
    loc['fap'] = fap
    loc['probabilities'] = probabilities
    loc['nyquist_factor'] = nyquist_factor

    return loc


def phase_plot(x, y, yerr, gp, fold_period, ylabel="y", t0=0, alpha=0.7, timesampling=0.001) :
    
    if t0 == 0:
        t0 = np.nanmin(x)

    phases, epochs = foldAt(x, fold_period, T0=t0, getEpoch=True)
    sortIndi = np.argsort(phases)
    min_epoch, max_epoch = int(np.nanmin(epochs)), int(np.nanmax(epochs))


    ti, tf = np.min(x), np.max(x)
    time = np.arange(ti, tf, timesampling)
    mphases = foldAt(time, fold_period, T0=t0)
    msortIndi = np.argsort(mphases)
    pred_mean, pred_var = gp.predict(y, time, return_var=True)
    pred_std = np.sqrt(pred_var)

    color = "#ff7f0e"
    plt.plot(mphases[msortIndi], pred_mean[msortIndi], "-", color=color, lw=2, alpha=0.5, label="GP model")
    #plt.fill_between(mphases[msortIndi], pred_mean[msortIndi]+pred_std[msortIndi], pred_mean[msortIndi]-pred_std[msortIndi], color=color, alpha=0.3, edgecolor="none")
    
    for ep in range(min_epoch, max_epoch+1) :
        inepoch = epochs[sortIndi] == ep
        if len(phases[sortIndi][inepoch]) :
            plt.errorbar(phases[sortIndi][inepoch],y[sortIndi][inepoch],yerr=yerr[sortIndi][inepoch], fmt='o', alpha=alpha, label="Cycle {}".format(ep))

    plt.ylabel(r"{}".format(ylabel), fontsize=16)
    plt.xlabel("phase (P={0:.3f} d)".format(fold_period), fontsize=16)
    plt.legend()
    plt.show()
