# Supplimentary Resources for the Foundation+YSE Analysis of the Cosmological Utility of SNe Ia in the z-band

If you make use of this repository or the analysis in Hayes+25, please cite: Hayes E.E., et al., 2025 ...

If you have any questions about this repository or the associated paper, you can direct them to Erin Hayes at eeh55@cam.ac.uk. Thank you!

## Data
The data for this project comes from the publicly available Young Supernova Experiment (YSE) Data Release 1 (DR1) and the Foundation Supernova Survey DR1. YSE DR1 is available on [Zenodo](https://zenodo.org/record/7317476) and Foundation DR1 is available on [Github](https://github.com/djones1040/Foundation_DR1/tree/master). The versions of the files used in this analysis can be found in the Data folder. If you make use of this data, please cite:

#### `YSE DR1`
* Aleo P.D., et al., 2023, [ApJS](https://iopscience.iop.org/article/10.3847/1538-4365/acbfba), [266, 9](https://ui.adsabs.harvard.edu/abs/2023ApJS..266....9A/abstract)

#### `Foundation DR1`
* Foley R.J., et al., 2018, [MNRAS](http://dx.doi.org/10.1093/mnras/stx3136), [475, 193](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475..193F/abstract)
* Jones D.O., et al., 2019, [ApJ](http://dx.doi.org/10.3847/1538-4357/ab2bec), [881, 19](https://ui.adsabs.harvard.edu/abs/2019ApJ...881...19J/abstract)

## Notebooks
In this folder, there are a set of jupyter notebooks that recreate versions of the figures in Hayes+24. You will also find a py script (`py/utils.py`) and a stylesheet (`stylesheet/zband.mplstyle`) to assist with plotting in the notebooks. 

The notebooks takes advantage of the data available in the Tables folder in this repository. You may adjust choices of the sample selection cuts, mass step location, or subset of data included in the BayeSN fits to see the effects on the RMS of the Hubble residuals and the size of the mass step.

## Tables
In this folder, you will find three tables:
* `full.csv`: a table with the necessary data to re-create the sample and results from Hayes+2024
  * `SNID`: the name of each object
  * `SURVEY`: if the object comes from YSE DR1 ("YSE") or Founation DR1 ("FOUNDATION")
  * `RA`: the right ascension of the object, in degrees
  * `DECL`: the declination of the object, in degrees
  * `SUBCLASS`: the subclass of SN Ia (i.e. 91bg, norm)
  * `REDSHIFT_FINAL`: the final, peculiar velocity corrected, redshift (sometimes also referred to as z_HD, for Hubble diagram, in other works)
  * `LOG_HOSTMASS(_ERR)`: the log stellar mass (and uncertainty) of the host galaxy associated with each SN
  * `LOG_HOSTSFR(_ERR)`: the log star formation rate (and uncertainty) of the host galaxy associated with each SN
  * `GRIZ_MU`: the distance modulus from the griz BayeSN fit
  * `GRIZ_MU_ERR`: the uncertainty on the distance modulus estimate from the griz BayeSN fit
  * `GRIZ_AV`: the dust extinction parameter from the griz BayeSN fit
  * `GRIZ_AV_ERR`: the uncertainty on the dust extinction estimate from the griz BayeSN fit
  * `GRIZ_THETA`: the light curve stretch parameter from the griz BayeSN fit
  * `GRIZ_THETA_ERR`: the uncertainty on the stretch estimate from the griz BayeSN fit
  * `GRIZ_T0`: the time of B-band maximum flux from the griz BayeSN fit, in MJD
  * `GRIZ_T0_ERR`: the uncertainty on the T0 estimate from the griz BayeSN fit, in MJD
  * `GRIZ_REDCHI2`: the reduced chi squared for the light curve fit from the griz BayeSN fit
  * `GRI_MU`: the distance modulus from the gri BayeSN fit
  * `GRI_MU_ERR`: the uncertainty on the distance modulus estimate from the gri BayeSN fit
  * `GRI_AV`: the dust extinction parameter from the gri BayeSN fit
  * `GRI_AV_ERR`: the uncertainty on the dust extinction estimate from the gri BayeSN fit
  * `GRI_THETA`: the light curve stretch parameter from the gri BayeSN fit
  * `GRI_THETA_ERR`: the uncertainty on the stretch estimate from the gri BayeSN fit
  * `GRI_T0`: the time of B-band maximum flux from the gri BayeSN fit, in MJD
  * `GRI_T0_ERR`: the uncertainty on the T0 estimate from the gri BayeSN fit, in MJD
  * `Z_MU`: the distance modulus from the z-band only BayeSN fit
  * `Z_MU_ERR`: the uncertainty on the distance modulus estimate from the z-band only BayeSN fit
  * `Z_THETA`: the stretch parameter from the z-band only BayeSN fit
  * `Z_THETA_ERR`: the uncertainty on the stretch parameter from the z-band only BayeSN fit
  * `Z_EXT`: the extinction in the z-band assuming Rv = 2.61 (from T21 model) and Av = GRI_AV
  * `N_G_OBS`: the number of g-band observations between (-10, 40) phase days
  * `FIRST_G_OBS`: the time of the first g-band observation between (-10, 40) phase days, in MJD
  * `N_R_OBS`: the number of r-band observations between (-10, 40) phase days
  * `FIRST_R_OBS`: the time of the first r-band observation between (-10, 40) phase days, in MJD
  * `N_I_OBS`: the number of i-band observations between (-10, 40) phase days
  * `FIRST_I_OBS`: the time of the first i-band observation between (-10, 40) phase days, in MJD
  * `N_Z_OBS`: the number of z-band observations between (-10, 40) phase days
  * `FIRST_Z_OBS`: the time of the first z-band observation between (-10, 40) phase days, in MJD
* `peak_mags_table.csv`: a table with the peak magnitude information used for the stretch-luminosity analysis in Hayes+2024
  * `SNID`: the name of each object
  * `SURVEY`: if the object comes from YSE DR1 ("YSE") or Founation DR1 ("FOUNDATION")
  * `REDSHIFT_FINAL`: the final, peculiar velocity corrected, redshift (sometimes also referred to as z_HD, for Hubble diagram, in other works)
  * `LOG_HOSTMASS`: the log stellar mass of the host galaxy associated with each SN
  * `PEAKMAG_X_DUSTCORR`: the peak dust-corrected, assuming Rv = 2.61 and Av = GRI_AV, apparent magnitude for filter X
  * `PEAKMAG_X_DUSTCORR_ERR`: the uncertainty on PEAKMAG_X_DUSTCORR for filter X
  * `PEAKMAG_X_DUSTCORR_THETACORR`: the peak dust-corrected, assuming Rv = 2.61 and Av = GRI_AV, and stretch-corrected magnitude for filter X
  * `PEAKMAG_X_DUSTCORR_THETACORR_ERR`: the uncertainty on PEAKMAG_X_DUSTCORR_THETACORR for filter X
  * `X_EXT`: the extinction in filter X assuming Rv = 2.61 and Av = GRI_AV
* `diff_Rv_Grayling.csv`: a table with the parameter estimates for the fits assuming different Rv values (Rv values from Grayling M., et al., 2024, [MNRAS](https://doi.org/10.1093/mnras/stae1202), [531, 953](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531..953G/abstract)); the columns have the same names as in `full.csv`, but with the additional prefixes of:
  * `LOWRV_`: indicates the parameter was fit assuming Rv = 2.39
  * `HIGHRV_`: indicates the parameter was fit assuming Rv = 3.14
* `other_surveys_comp_data.csv`: a table with the redshifts and host galaxy masses for the iPTF, CSP, and CfA samples used in other similar analyses of SNe Ia in the NIR; used to create Figures 1 and 2 in Hayes+2024

These three tables have all the necessary information to reproduce the figures and results from the paper.

## Figures
In this folder, you will find full resolution copies of the figures from Hayes+24 and plots of the BayeSN fits to the data.
