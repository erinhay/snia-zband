# Supplimentary Resources for the Foundation+YSE Analysis of the Cosmological Utility of SNe Ia in the z-band

If you make use of this repository or the analysis in Hayes+25, please cite: Hayes E.E., et al., 2025, [pre-print](
https://doi.org/10.48550/arXiv.2506.04309).

If you have any questions about this repository or the associated paper, you can direct them to Erin Hayes at eeh55@cam.ac.uk. Thank you!

## Data
The data for this project comes from the publicly available Young Supernova Experiment (YSE) Data Release 1 (DR1) and the Foundation Supernova Survey DR1. YSE DR1 is available on [Zenodo](https://zenodo.org/record/7317476) and Foundation DR1 is available on [Github](https://github.com/djones1040/Foundation_DR1/tree/master). The versions of the files used in this analysis can be found in the Data folder. If you make use of this data, please cite:

#### `YSE DR1`
* Aleo P.D., et al., 2023, [ApJS](https://iopscience.iop.org/article/10.3847/1538-4365/acbfba), [266, 9](https://ui.adsabs.harvard.edu/abs/2023ApJS..266....9A/abstract)

#### `Foundation DR1`
* Foley R.J., et al., 2018, [MNRAS](http://dx.doi.org/10.1093/mnras/stx3136), [475, 193](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475..193F/abstract)
* Jones D.O., et al., 2019, [ApJ](http://dx.doi.org/10.3847/1538-4357/ab2bec), [881, 19](https://ui.adsabs.harvard.edu/abs/2019ApJ...881...19J/abstract)

## Notebooks
In this folder, there are a set of jupyter notebooks that recreate versions of the figures in Hayes+25. You will also find a py script (`py/utils.py`) and a stylesheet (`stylesheet/zband.mplstyle`) to assist with plotting in the notebooks. 

The notebooks takes advantage of the data available in the Tables folder in this repository. You may adjust choices of the sample selection cuts, mass step location, or subset of data included in the BayeSN fits to see the effects on the RMS of the Hubble residuals and the size of the mass step.

## Tables
In this folder, you will find two tables:
* `mnras_table.txt`: a table with the necessary data to re-create the sample and results from Hayes+25
  * `SNID`: the name of each object
  * `SURVEY`: if the object comes from YSE DR1 ("YSE") or Founation DR1 ("FOUNDATION")
  * `RA`: the right ascension of the object, in degrees
  * `DECL`: the declination of the object, in degrees
  * `SUBCLASS`: the subclass of SN Ia (i.e. 91bg, norm)
  * `REDSHIFT_FINAL`: the final, peculiar velocity corrected, redshift (sometimes also referred to as z_HD, for Hubble diagram, in other works)
  * `LOG_HOSTMASS(_ERR)`: the log stellar mass (and uncertainty) of the host galaxy associated with each SN
  * `LOG_HOSTSFR(_ERR)`: the log star formation rate (and uncertainty) of the host galaxy associated with each SN
  * `GRIZ_MU(_ERR)`: the distance modulus (and uncertainty) from the griz BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `GRIZ_AV(_ERR)`: the dust extinction parameter (and uncertainty) from the griz BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `GRIZ_THETA(_ERR)`: the light curve stretch parameter (and uncertainty) from the griz BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `GRIZ_T0(_ERR)`: the time of B-band maximum flux (and uncertainty) from the griz BayeSN fit, in MJD assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `GRIZ_REDCHI2`: the reduced chi squared for the light curve fit from the griz BayeSN fit assuming Rv = 2.61
  * `GRI_MU(_ERR)`: the distance modulus (and uncertainty) from the gri BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `GRI_AV(_ERR)`: the dust extinction parameter (and uncertainty) from the gri BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `GRI_THETA(_ERR)`: the light curve stretch parameter (and uncertainty) from the gri BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `GRI_T0(_ERR)`: the time of B-band maximum flux (and uncertainty) from the gri BayeSN fit, in MJD assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `Z_MU(_ERR)`: the distance modulus (and uncertainty) from the z-band only BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `Z_THETA(_ERR)`: the stretch parameter (and uncertainty) from the z-band only BayeSN fit assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `X_MU(_ERR)`: the distance modulus (and uncertainty) from the X-band only BayeSN fit, where X may be 'G', 'R', or 'I' assuming Rv = 2.61
    * `LOWRV_`: prefix indicates the parameter was fit assuming Rv = 2.39
    * `HIGHRV_`: prefix indicates the parameter was fit assuming Rv = 3.14
  * `PEAKMAG_X_DUSTCORR`: the peak dust-corrected, assuming Rv = 2.61 and Av = GRIZ_AV, apparent magnitude for filter X
  * `PEAKMAG_X_DUSTCORR_ERR`: the uncertainty on PEAKMAG_X_DUSTCORR for filter X
  * `PEAKMAG_X_DUSTCORR_THETACORR`: the peak dust-corrected, assuming Rv = 2.61 and Av = GRIZ_AV, and stretch-corrected magnitude for filter X
  * `PEAKMAG_X_DUSTCORR_THETACORR_ERR`: the uncertainty on PEAKMAG_X_DUSTCORR_THETACORR for filter X
  * `X_EXT`: the extinction in filter X assuming Rv = 2.61 and Av = GRIZ_AV
  * `N_X_OBS`: the number of X-band observations between (-10, 40) phase days
  * `FIRST_X_OBS`: the time of the first X-band observation between (-10, 40) phase days, in MJD
* `other_surveys_comp_data.csv`: a table with the redshifts and host galaxy masses for the iPTF, CSP, and CfA samples used in other similar analyses of SNe Ia in the NIR; used to create Figures 1 and 2 in Hayes+2025

These two tables have all the necessary information to reproduce the figures and results from the paper. Note that the Rv values for the split Rv case come from Grayling M., et al., 2024, [MNRAS](https://doi.org/10.1093/mnras/stae1202), [531, 953](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531..953G/abstract).

## Figures
In this folder, you will find full resolution copies of the figures from Hayes+25 and plots of the BayeSN fits to the data.
