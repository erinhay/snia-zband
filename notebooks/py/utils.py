import numpy as np
import matplotlib.pyplot as plt
import astropy.table as at
import astropy.constants as ac
from astropy.cosmology import FlatLambdaCDM
import extinction

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
from jax import random
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

try:
    from bayesn import SEDmodel
except:
    pass


wave_dict = {'g': 4810, 'r': 6170, 'i': 7520, 'z': 8600} # in AA
blue = '#529dd1'
green = 'green'
red = 'crimson'

def heliocorr(zhel, RA, Dec):
    """
    A function to transform redshifts from the heliocentric-frame to the
    CMB-frame using Planck (2018) CMB Dipole measurements.

    Inputs:
        zhel: float or numpy array, input heliocentric redshift(s)
        RA: float or numpy array, object equatorial right ascension(s)
        Dec: float or numpy array, object equatorial declination(s)

    Outputs:
        zcmb: numpy array, redshift(s) corrected to the CMB frame
        alpha: float or numpy array, angular separation from CMB dipole (rad)
    """

    v_Sun_Planck = 369.82  # +/- 0.11 km/s
    l_dipole_Planck = 264.021  # +/- 0.011 deg
    b_dipole_Planck = 48.253  # +/- 0.005 deg
    c = 299792.458  # km/s

    # Co-ords of North Galactic Pole (ICRS): RA = 192.729 ± 0.035 deg, Dec = 27.084 ± 0.023 deg
    # (https://doi.org/10.1093/mnras/stw2772)
    # Co-ords of Galactic Centre (ICRS): RA = 17h45m40.0409s, Dec = −29d00m28.118s (see above reference)
    #                                    RA = 266.41683708 deg, Dec = -29.00781056 deg
    # Ascending node of the galactic plane = arccos(sin(Dec_GC)*cos(Dec_NGP)-cos(Dec_GC)*sin(Dec_NGP)*cos(RA_NGP-RA_GC))
    #                                      = 122.92828126730255 = l_0
    # Transform CMB dipole from (l,b) to (RA,Dec):
    #     Dec = arcsin(sin(Dec_NGP)*sin(b)+cos(Dec_NGP)*cos(b)*cos(l_0-l))
    #         = -6.9895105228347 deg
    #     RA = RA_NGP + arctan((cos(b)*sin(l_0-l)) / (cos(Dec_NGP)*sin(b)-sin(Dec_NGP)*cos(b)*cos(l_0-l)))
    #        = 167.81671014708002 deg

    # Astropy co-ordinates are old and low precision:
    # RA_NGP_J2000 = 192.8594812065348, Dec_NGP_J2000 = 27.12825118085622, which are converted from B1950
    # RA_NGP_B1950 = 192.25, Dec_NGP_B1950 = 27.4
    # l_0_B1950 = 123
    # l_0_J2000 = 122.9319185680026
    # Introduces around 1e-6 error in redshift

    RA_Sun_Planck = 167.816710  # deg
    Dec_Sun_Planck = -6.989510  # deg

    rad = np.pi / 180.0
    # using Vincenty formula because it is more accurate
    alpha = np.arctan2(
        np.hypot(
            np.cos(Dec_Sun_Planck * rad) * np.sin(np.fabs(RA - RA_Sun_Planck) * rad),
            np.cos(Dec * rad) * np.sin(Dec_Sun_Planck * rad)
            - np.sin(Dec * rad)
            * np.cos(Dec_Sun_Planck * rad)
            * np.cos(np.fabs(RA - RA_Sun_Planck) * rad),
        ),
        np.sin(Dec * rad) * np.sin(Dec_Sun_Planck * rad)
        + np.cos(Dec * rad)
        * np.cos(Dec_Sun_Planck * rad)
        * np.cos(np.fabs(RA - RA_Sun_Planck) * rad),
    )

    v_Sun_proj = v_Sun_Planck * np.cos(alpha)

    z_Sun = np.sqrt((1.0 + (-v_Sun_proj) / c) / (1.0 - (-v_Sun_proj) / c)) - 1.0
    # Full special rel. correction since it is a peculiar vel

    min_z = 0.0

    zcmb = np.where(zhel > min_z, (1 + zhel) / (1 + z_Sun) - 1, zhel)

    return zcmb, alpha



def get_peculiar_velocity_unc(z_HD, sigma_z, sigma_pec=150):
    """
    Calculates the uncertainty in peculiar velocity based on redshift and uncertainty in distance.

    Parameters:
    - z_HD (float): Redshift of the source.
    - sigma_pec (float, optional): Uncertainty in peculiar velocity in km/s. Default is 150 km/s.

    Returns:
    - peculiar_velocity_unc (float): Uncertainty in peculiar velocity.
    """
    
    # Speed of light in kilometers per second
    speed_of_light = ac.c.value / 1000  # km/s
    
    # Formula to calculate uncertainty in peculiar velocity
    peculiar_velocity_unc = np.sqrt( ( (sigma_pec/speed_of_light)**2 + sigma_z**2 ) * (5 / (np.log(10) * z_HD))**2)
    
    return peculiar_velocity_unc  # Return the calculated uncertainty



def model_for_plotting(zs, H0=73.24, Omega_m=0.28):
    """
    Generates model data for plotting distance modulus against redshift.

    Parameters:
    - zs (array-like): Array of redshift values.
    - H0 (float, optional): Hubble constant in km/s/Mpc. Default is 73.24 km/s/Mpc.
    - Omega_m (float, optional): Matter density parameter. Default is 0.28.

    Returns:
    - zs (array-like): Array of redshift values.
    - model (array-like): Array of model distance modulus values corresponding to input redshifts.
    """

    # Speed of light in kilometers per second
    speed_of_light = ac.c.value / 1000  # km/s

    # Creating a cosmological model with specified parameters
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

    # Calculating dimensionless luminosity distance and converting to Mpc
    dimensionless_lumdist = cosmo.luminosity_distance(zs).value / cosmo.hubble_distance.value

    # Calculating distance modulus using the model equation
    model = 5 * np.log10((speed_of_light / H0) * dimensionless_lumdist) + 25

    # Returning redshifts and corresponding model distance modulus values
    return zs, model



def plot_single_bayesn_fit(snid, meta, lc, chains = None, color_dict={'G': green, 'R': red, 'I': 'tab:purple', 'Z': 'tab:orange'}):

    chains = np.load(chains, allow_pickle=True).item()
    times = np.arange(-10, 40+0.5, 0.5)
    model = SEDmodel(load_model='T21_model')

    fig, ax = plt.subplots(4, 2, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1, 3, 1]}, sharex=True, sharey='row')
    ax[0,0].invert_yaxis()
    ax[2,0].invert_yaxis()

    for f1, flt_set in enumerate([['G', 'R'], ['I', 'Z']]):
        for f2, flt in enumerate(flt_set):
            single_flt = lc[lc['FLT'] == flt.lower()+'_PS1']
            appended_times = np.concatenate([times, single_flt['PHASE'].value])
            
            flux_grid = model.get_flux_from_chains(appended_times, [flt.lower()+'_PS1'], chains, [meta['REDSHIFT_FINAL']], [meta['MWEBV']], mag=True)
            mags_mean, mags_unc = flux_grid.mean(axis=(0, 1)), flux_grid.std(axis=(0, 1))

            _, _, bars = ax[f1*2,f2].errorbar(single_flt['PHASE'], single_flt['MAG'], yerr=single_flt['MAGERR'],
                                              color=color_dict[flt], ls='None', marker='o', label=flt.lower(), zorder=3)
            [bar.set_alpha(0.3) for bar in bars]

            ax[f1*2,f2].plot(appended_times[:len(times)], mags_mean[0][:len(times)], color='dimgray', lw=2, zorder=2)
            ax[f1*2,f2].fill_between(appended_times[:len(times)], mags_mean[0][:len(times)]-mags_unc[0][:len(times)],
                                     mags_mean[0][:len(times)]+mags_unc[0][:len(times)],
                                     color='dimgray', alpha=0.3, zorder=1)

            ax[(f1*2)+1,f2].axhline(0, 0, 1, color='dimgray', ls='--', zorder=1)
            _, _, bars = ax[(f1*2)+1,f2].errorbar(appended_times[len(times):], single_flt['MAG'] - mags_mean[0][len(times):],
                                                  yerr=np.sqrt(single_flt['MAGERR']**2 + mags_unc[0][len(times):]**2),
                                                  color=color_dict[flt], ls='None', marker='o', zorder=2)
            [bar.set_alpha(0.3) for bar in bars]

            ax[(f1*2)+1,f2].set_xlabel('Phase [days]')
            ax[f1*2,f2].legend()
        ax[f1*2,0].set_ylabel('Magnitude')
        ax[(f1*2)+1,0].set_ylabel('Residual')

    ax[1,0].set_ylim([-1*np.max(ax[1,0].get_ylim()), np.max(ax[1,0].get_ylim())])
    ax[3,0].set_ylim([-1*np.max(ax[3,0].get_ylim()), np.max(ax[3,0].get_ylim())])

    fig.suptitle('SN '+snid, fontsize=24, x=0.55)
    fig.align_labels()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    return fig, ax



def plot_multi_bayesn_fits(snid, meta, lc, griz_chains = None, gri_chains = None, z_chains = None):

    model = SEDmodel(load_model='T21_model')

    fig, ax = plt.subplots(4, 2, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1, 3, 1]}, sharex=True, sharey='row')
    ax[0,0].invert_yaxis()
    ax[2,0].invert_yaxis()

    times = np.arange(-10, 40+0.5, 0.5)
    color_dict_fitted = {'GRIZ': blue, 'GRI': green, 'Z': red}
    ls_dict = {'griz': '-', 'gri': '--', 'z': '-.'}
    
    chains_tags = {}
    if griz_chains:
        chains_tags['griz'] = griz_chains
    if gri_chains:
        chains_tags['gri'] = gri_chains
    if z_chains:
        chains_tags['z'] = z_chains

    for f1, flt_set in enumerate([['G', 'R'], ['I', 'Z']]):
        for f2, flt in enumerate(flt_set):
            single_flt = lc[lc['FLT'] == flt.lower()+'_PS1']

            _, _, bars = ax[f1*2,f2].errorbar(single_flt['PHASE'], single_flt['MAG'], yerr=single_flt['MAGERR'],
                                color='k', ls='None', marker='o', label=flt.lower(), zorder=3)
            [bar.set_alpha(0.3) for bar in bars]
            ax[(f1*2)+1,f2].axhline(0, 0, 1, color='dimgray', ls='--', zorder=1)
            ax[(f1*2)+1,f2].set_xlabel('Phase [days]')
        ax[f1*2,0].set_ylabel('Mag.')
        ax[(f1*2)+1,0].set_ylabel('Resid.')

    for chains_tag in chains_tags.keys():
        chains = np.load(chains_tags[chains_tag], allow_pickle=True).item()
    
        for f1, flt_set in enumerate([['G', 'R'], ['I', 'Z']]):
            for f2, flt in enumerate(flt_set):

                if not flt.lower() in chains_tag:
                    continue

                single_flt = lc[lc['FLT'] == flt.lower()+'_PS1']
                appended_times = np.concatenate([times, single_flt['PHASE'].value])

                flux_grid = model.get_flux_from_chains(appended_times, [flt.lower()+'_PS1'], chains, [meta['REDSHIFT_FINAL']], [meta['MWEBV']], mag=True)
                mags_mean, mags_unc = flux_grid.mean(axis=(0, 1)), flux_grid.std(axis=(0, 1))

                single_flt = lc[lc['FLT'] == flt.lower()+'_PS1']

                ax[f1*2,f2].plot(appended_times[:len(times)], mags_mean[0][:len(times)], color=color_dict_fitted[chains_tag.upper()],
                                 lw=2, zorder=2, ls=ls_dict[chains_tag], label=chains_tag+' Fit')
                ax[f1*2,f2].fill_between(appended_times[:len(times)], mags_mean[0][:len(times)]-mags_unc[0][:len(times)],
                                         mags_mean[0][:len(times)]+mags_unc[0][:len(times)],
                                         color=color_dict_fitted[chains_tag.upper()], alpha=0.3, zorder=1)
                
                _, _, bars = ax[(f1*2)+1,f2].errorbar(appended_times[len(times):], single_flt['MAG'] - mags_mean[0][len(times):],
                                                      yerr=np.sqrt(single_flt['MAGERR']**2 + mags_unc[0][len(times):]**2), 
                                                      color=color_dict_fitted[chains_tag.upper()],
                                                      ls='None', marker='o', zorder=2)
                [bar.set_alpha(0.3) for bar in bars]
                ax[f1*2,f2].legend()

    ax[1,0].set_ylim([-1*np.max(ax[1,0].get_ylim()), np.max(ax[1,0].get_ylim())])
    ax[3,0].set_ylim([-1*np.max(ax[3,0].get_ylim()), np.max(ax[3,0].get_ylim())])

    fig.suptitle('SN '+snid, fontsize=24, x=0.55)
    fig.align_labels()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    return fig, ax



def plot_histogram(data, key, bins=None, split_by_survey=False):
    """
    Plots histogram.

    Parameters:
    - data (DataFrame): DataFrame containing the data to be plotted.
    - key (str): The key indicating which data column to plot. Must be one of: 'REDSHIFT_FINAL', 'LOG_HOSTMASS', or contain 'THETA' or 'AV'.
    - bins (array-like or int, optional): Bins for histogram. If key is not in supported list, then bins must be specified.
    - bins (bool, optional): Whether to plot the 'THETA' or 'AV' distributions by survey.

    Returns:
    - fig (matplotlib.figure.Figure): The generated matplotlib figure.
    """
    survey_mask = np.repeat(True, len(data))
    fig = plt.figure()

    # Setting up bins and labels based on the key
    if key == 'LOG_HOSTMASS':
        bins = np.arange(6, 13, 0.5)
        plt.xlabel('Log$_{10}$(M$_{*}$/M$_{\odot}$)')
        loc = 'upper left'

    elif key == 'REDSHIFT_FINAL':
        bins = np.arange(0, 0.16, 0.01)
        plt.xlabel('Redshift')
        loc = 'upper right'

    elif 'THETA' in key:
        if split_by_survey:
            survey_mask = data['SURVEY'] == 'YSE'
        bins = np.arange(-2, 4, 0.5)
        plt.xlabel('$\\theta$')
        loc = 'upper right'
    
    elif 'AV' in key:
        if split_by_survey:
            survey_mask = data['SURVEY'] == 'YSE'
        bins = np.arange(0, 1.2, 0.1)
        plt.xlabel('$A_{V}$')
        loc = 'upper right'

    # Adding histogram data for other surveys if key corresponds to 'LOG_HOSTMASS' or 'REDSHIFT_FINAL'
    if np.isin(key, ['LOG_HOSTMASS', 'REDSHIFT_FINAL']):
        other_survey_data = at.Table.read('../tables/other_surveys_comp_data.csv', format='csv')
        other_survey_data = other_survey_data[other_survey_data[key] > -9998.]

        # Histograms for each survey
        plt.hist(other_survey_data[other_survey_data['SURVEY'] == 'CSP'][key], bins=bins, color='k', histtype='step', lw=2, label='CSP', zorder=2)
        plt.hist(other_survey_data[other_survey_data['SURVEY'] == 'iPTF'][key], bins=bins, color='k', ls='--', histtype='step', lw=2, label='iPTF', zorder=2)
        plt.hist(other_survey_data[other_survey_data['SURVEY'] == 'CfA'][key], bins=bins, color='k', ls='-.', histtype='step', lw=2, label='CfA', zorder=2)

    # Histogram for YSE+Foundation data
    if split_by_survey:
        plt.hist(data[key][~survey_mask], bins=bins, color=blue, alpha=0.5, label='Foundation', zorder=1)
        plt.hist(data[key][~survey_mask], bins=bins, color=blue, histtype='step', lw=2, zorder=1)

        plt.hist(data[key][survey_mask], bins=bins, color=green, alpha=0.5, label='YSE', zorder=1)
        plt.hist(data[key][survey_mask], bins=bins, color=green, histtype='step', lw=2, zorder=1)

    else:
        plt.hist(data[key], bins=bins, color=blue, alpha=0.5, label='YSE+Foundation', zorder=1)
        plt.hist(data[key], bins=bins, color=blue, histtype='step', lw=2, zorder=1)

    # Adding labels, legend, and adjusting layout
    plt.ylabel('Number of SNe Ia')
    plt.legend(loc=loc, fontsize=14)
    plt.tight_layout()

    return fig



def apply_data_cuts(data, redshift_lower_limit=0.015, redshift_upper_limit_YSE=0.1, redshift_upper_limit_Foundation=0.08, keep_91Ts=False, g_obs=1, r_obs=1, i_obs=1, z_obs=3, gri_prepeak=True, z_prepeak=False, chi_squared_threshold=3, t0_unc_threshold=1, t0_agreement_threshold=0.5, Av_upper_limit=1, theta_lower_limit=-1.5, theta_upper_limit=3, host_galaxy_mass=True, save_path='', print_summary=True):
    """
    Applies data cuts to the input dataset based on various criteria.

    Parameters:
    - data (DataFrame): The dataset to apply data cuts to.
    - redshift_lower_limit (float, optional): Lower limit for redshift. Default is 0.01.
    - redshift_upper_limit_YSE (float, optional): Upper limit for redshift for YSE survey. Default is 0.1.
    - redshift_upper_limit_Foundation (float, optional): Upper limit for redshift for Foundation survey. Default is 0.08.
    - keep_91Ts (bool, optional): Whether to keep SN Ia-91T in the dataset. Default is True.
    - g_obs (int, optional): Minimum number of observations in the g-band. Default is 1.
    - r_obs (int, optional): Minimum number of observations in the r-band. Default is 1.
    - i_obs (int, optional): Minimum number of observations in the i-band. Default is 1.
    - z_obs (int, optional): Minimum number of observations in the z-band. Default is 4.
    - gri_prepeak (bool, optional): Whether to require at least 1 observation before peak in one of the g-, r-, or i-bands. Default is True.
    - z_prepeak (bool, optional): Whether to require z-band pre-peak data. Default is False.
    - chi_squared_threshold (float, optional): Maximum reduced chi-squared for the fit to the full griz light curve. Default is 5.
    - t0_unc_threshold (float, optional): Maximum uncertainty on the time of B-band maximum flux for the fit to the griz light curve and the gri light curve. Default is 1.
    - t0_agreement_threshold (float, optional): Maximum allowed disagreement in the value of the time of B-band maximum flux as estimated from the griz light curve fit and that from the gri light curve fit.
    - Av_upper_limit (float, optional): Maximum Av (extinction) value. Default is 1.
    - theta_lower_limit (float, optional): Lower limit for theta. Default is -1.5.
    - theta_upper_limit (float, optional): Upper limit for theta. Default is 3.
    - host_galaxy_mass (bool, optional): Whether to require a host mass estimate. Default is True.
    - save_path (str, optional): If specified, save the list of names of the objects in the sample to the tables folder under this name. Default does not save the sample.
    - print_summary (bool, optional): Print summary of data cuts. Default is True.

    Returns:
    - sample (DataFrame): The dataset after applying data cuts.
    """
    sample = data
    if print_summary:
        print(f"Total Spec Sample:               {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Remove abnormal SNe Ia (i.e. SN Ic, SN Ia-91bg; optionally SN Ia-91T)
    sample = sample[np.isin(sample['SUBCLASS'], ['SNIa-norm', 'SNIa-91T-like'])]
    if not keep_91Ts:
        sample = sample[sample['SUBCLASS'] == 'SNIa-norm']
    if print_summary:
        print(f"Normal Ias:                      {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Redshift requirement
    sample = sample[sample['REDSHIFT_FINAL'] > redshift_lower_limit]

    if redshift_upper_limit_YSE < redshift_upper_limit_Foundation:
        sample = sample[sample['REDSHIFT_FINAL'] < redshift_upper_limit_Foundation]
        sample = sample[np.logical_or(sample['SURVEY'] == 'FOUNDATION', sample['REDSHIFT_FINAL'] < redshift_upper_limit_YSE)]
    elif redshift_upper_limit_Foundation < redshift_upper_limit_YSE:
        sample = sample[sample['REDSHIFT_FINAL'] < redshift_upper_limit_YSE]
        sample = sample[np.logical_or(sample['SURVEY'] == 'YSE', sample['REDSHIFT_FINAL'] < redshift_upper_limit_Foundation)]
    else:
        sample = sample[sample['REDSHIFT_FINAL'] < redshift_upper_limit_YSE]

    if print_summary:
        print(f"{redshift_lower_limit} < z < {np.max([redshift_upper_limit_YSE, redshift_upper_limit_Foundation])}:                 {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    if redshift_upper_limit_YSE < redshift_upper_limit_Foundation and print_summary:
        print(f'(z < {redshift_upper_limit_YSE} for YSE')
    elif redshift_upper_limit_Foundation < redshift_upper_limit_YSE and print_summary:
        print(f'(z < {redshift_upper_limit_Foundation} for Foundation)')

    # Require at least X_obs data point in each of the g-, r-, and i-bands
    sample = sample[(sample['N_G_OBS'] >= g_obs) * (sample['N_R_OBS'] >= r_obs) * (sample['N_I_OBS'] >= i_obs)]

    # if gri_prepeak == True, require at least 1 observation before peak in one of the g-, r-, or i-bands
    if gri_prepeak:
        sample = sample[np.logical_or(np.logical_or(sample['FIRST_G_OBS'] < sample['GRIZ_T0'], sample['FIRST_R_OBS'] < sample['GRIZ_T0']),
                                      sample['FIRST_I_OBS'] < sample['GRIZ_T0'])]
    if print_summary:
        print(f"gri Data Requirement:            {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Require at least z_obs data point in the z-band
    sample = sample[sample['N_Z_OBS'] >= z_obs]

    ## if z_prepeak, require z-band pre-peak data
    if z_prepeak:
        sample = sample[sample['FIRST_Z_OBS'] <= sample['GRIZ_T0']]

    if print_summary:
        print(f"z-band Data Requirement:         {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Require a reduced chi-squared of less than 5 for the fit the full griz light curve
    sample = sample[sample['GRIZ_REDCHI2'] < chi_squared_threshold]
    if print_summary:
        print(f"Reduced Chi Squared < {chi_squared_threshold}:         {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Require uncertainties on the time of B-band maximum flux of less than some threshold for both the fit to the griz light curve and the gri light curve
    sample = sample[np.logical_and(sample['GRIZ_T0_ERR'] < t0_unc_threshold, sample['GRI_T0_ERR'] < t0_unc_threshold)]
    sample = sample[np.abs(sample['GRIZ_T0'] - sample['GRI_T0']) < t0_agreement_threshold]
    if print_summary:
        print(f"T0 Cut:                          {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Av requirement
    sample = sample[sample['GRIZ_AV'] < Av_upper_limit]
    if print_summary:
        print(f"Av < {Av_upper_limit}:                          {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Require theta within the bounds of the T21 training sample
    sample = sample[np.logical_and(sample['GRIZ_THETA'] < theta_upper_limit, sample['GRIZ_THETA'] > theta_lower_limit)]
    if print_summary:
        print(f"{theta_lower_limit} < theta < {theta_upper_limit}:                {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")

    # Require a host mass estimate
    if host_galaxy_mass:
        sample = sample[sample['LOG_HOSTMASS'] >= 0]
        if print_summary:
            print(f"Host Mass Estimate:              {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")
    
    if print_summary:
        print(f"\nFinal Sample:                    {len(sample)} (YSE: {len(sample[sample['SURVEY'] == 'YSE'])}, Foundation: {len(sample[sample['SURVEY'] == 'FOUNDATION'])})")
        print("* For complete description of the default data cuts, see Hayes+2024.")

    if len(save_path) > 0:
        np.savetxt(f'../tables/{save_path}.csv', list(sample['SNID'].value), fmt='%s')

    return sample



def plot_hubble_diagram(data, fit_filters = 'griz', dust_corrected=True, colors={'YSE': green, 'Foundation': blue}, markers={'YSE': 's', 'Foundation': 'o'}):
    """
    Plots Hubble diagram for BayeSN fits to specified subset of data.
    """

    resids = []

    fig, ax = plt.subplots(2, 1, figsize=(6,6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    zs, model = model_for_plotting(np.linspace(np.min(data['REDSHIFT_FINAL'])-0.002, np.max(data['REDSHIFT_FINAL'])+0.008, 50))
    ax[0].plot(zs, model, color='dimgray')
    ax[1].axhline(0, xmin=0, xmax=1, ls='-', c='dimgray')

    for survey in ['YSE', 'Foundation']:
        subset = data[data['SURVEY'] == survey.upper()]

        redshift_final, mu_LCDM = model_for_plotting(subset['REDSHIFT_FINAL'])
        mu = subset[fit_filters.upper()+'_MU']
        mu_err = subset[fit_filters.upper()+'_MU_ERR']
        sigma_z = subset['SIGMA_Z']

        propagated_redshift_measurement_unc = get_peculiar_velocity_unc(redshift_final, subset['SIGMA_Z'], sigma_pec=0)
        resid_err = np.sqrt(subset[fit_filters.upper()+'_MU_ERR']**2 + propagated_redshift_measurement_unc**2)

        if dust_corrected and len(fit_filters) == 1:
            wave = wave_dict[fit_filters]
            ext_mag = np.zeros(len(subset))
            for r, row in enumerate(subset):
                ext_mag[r] = extinction.fitzpatrick99(np.array([wave]), a_v = row['GRI_AV'], r_v=2.610)

            mu = mu - ext_mag
        
        resid = mu - mu_LCDM
        resids += list(resid)

        _, _, bars = ax[0].errorbar(redshift_final, mu, xerr=sigma_z, yerr=mu_err, color=colors[survey], ls='None', marker=markers[survey], ms=5, label=survey)
        [bar.set_alpha(0.25) for bar in bars]

        _, _, bars = ax[1].errorbar(redshift_final, resid, yerr=resid_err, color=colors[survey], ls='None', marker=markers[survey], ms=5)
        [bar.set_alpha(0.25) for bar in bars]

    xmin, xmax = ax[1].get_xlim()

    model_PV_unc = get_peculiar_velocity_unc(zs, 0., sigma_pec=150)
    ax[1].plot(zs, model_PV_unc, ls='--', color='dimgray')
    ax[1].plot(zs, -model_PV_unc, ls='--', color='dimgray')
    ax[1].set_xlim(xmin, xmax)
    ax[1].set_ylim(-0.8, 0.8)

    rms = np.sqrt(np.mean(np.array(resids)**2))
    xloc = ((np.max(ax[0].get_xlim()) - np.min(ax[0].get_xlim()))* 0.6) + np.min(ax[0].get_xlim())
    yloc = ((np.max(ax[0].get_ylim()) - np.min(ax[0].get_ylim()))* 0.1) + np.min(ax[0].get_ylim())
    ax[0].text(xloc, yloc, 'RMS = '+str(round(rms, 3)), fontsize=14)

    ax[1].set_xlabel('$z$')
    ax[0].set_ylabel('$\\mu$')
    ax[1].set_ylabel('$\\mu - \\mu_{\\Lambda \\rm{CDM}}(z)$')
    ax[0].set_title('$'+fit_filters+'$ Hubble Diagram')
    ax[0].legend(fontsize=12)
    fig.align_labels()
    fig.subplots_adjust(hspace=0, wspace=0)

    return fig, ax



def compute_intrinsic_abs_magnitude(y, yerr, survey_mask=None, hmc_kwargs={}):

    def model(y, yerr, survey_mask):
        mu = numpyro.sample("M_0_int", dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
        
        if survey_mask is None:
            sigma_2 = numpyro.sample("sigma_2_res", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
        else:
            sigma_2_YSE = numpyro.sample("sigma_2_res_YSE", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_Foundation = numpyro.sample("sigma_2_res_Foundation", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2 = sigma_2_YSE*survey_mask + sigma_2_Foundation*(1-survey_mask)
            
        numpyro.sample("obs", dist.Normal( mu, jnp.sqrt(sigma_2 + yerr**2)), obs=y)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    kernel = NUTS(model)
    num_warmup = hmc_kwargs.pop('num_warmup', 4000)
    num_samples = hmc_kwargs.pop('num_samples', 10_000)
    sampler = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, **hmc_kwargs)
    sampler.run(rng_key_, y, yerr, survey_mask)

    return sampler
    


def compute_sigma_res(y, yerr, survey_mask=None, hmc_kwargs={}):

    def model(y, yerr, survey_mask):
        if survey_mask is None:
            sigma_2 = numpyro.sample("sigma_2_res", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
        else:
            sigma_2_YSE = numpyro.sample("sigma_2_res_YSE", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_Foundation = numpyro.sample("sigma_2_res_Foundation", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2 = sigma_2_YSE*survey_mask + sigma_2_Foundation*(1-survey_mask)
            
        numpyro.sample("obs", dist.Normal(0, jnp.sqrt(sigma_2 + yerr**2)), obs=y)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    kernel = NUTS(model)
    num_warmup = hmc_kwargs.pop('num_warmup', 4000)
    num_samples = hmc_kwargs.pop('num_samples', 10_000)
    sampler = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, **hmc_kwargs)
    sampler.run(rng_key_, y, yerr, survey_mask)

    return sampler



def compute_stretch_lum_relation(x, xerr, y, yerr, survey_mask=None, hmc_kwargs={}):

    def model(x, xerr, y, yerr, survey_mask):
        a = numpyro.sample("a", dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
        b = numpyro.sample("b", dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
        
        if survey_mask is None:
            sigma_2_res = numpyro.sample("sigma_2_res", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
        else:
            sigma_2_YSE = numpyro.sample("sigma_2_res_YSE", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_Foundation = numpyro.sample("sigma_2_res_Foundation", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_res = sigma_2_YSE*survey_mask + sigma_2_Foundation*(1-survey_mask)
        
        with numpyro.plate('data', len(x)):
            true_x = numpyro.sample("true_x", dist.Normal(x, xerr))
            numpyro.sample("y", dist.Normal( (a*true_x) + b, jnp.sqrt(sigma_2_res + yerr**2)), obs=y)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    kernel = NUTS(model)
    num_warmup = hmc_kwargs.pop('num_warmup', 4000)
    num_samples = hmc_kwargs.pop('num_samples', 10_000)
    sampler = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, **hmc_kwargs)
    sampler.run(rng_key_, x, xerr, y, yerr, survey_mask)

    return sampler



def plot_stretch_lum_relation(x, xerr, y, yerr, survey_mask=None, samples=None):
    fig, ax = plt.subplots(1, 1, figsize=(8,4), sharex=True, sharey=True)
    ax.invert_yaxis()

    _, _, bars = ax.errorbar(x[~survey_mask], y[~survey_mask], xerr=xerr[~survey_mask], yerr=yerr[~survey_mask],
                             color=blue, ls='None', marker='o', ms=5, label='Foundation')
    [bar.set_alpha(0.25) for bar in bars]
    _, _, bars = ax.errorbar(x[survey_mask], y[survey_mask], xerr=xerr[survey_mask], yerr=yerr[survey_mask],
                             color=green, ls='None', marker='s', ms=5, label='YSE')
    [bar.set_alpha(0.25) for bar in bars]

    if not samples == None:
        thetas = np.arange(-1.55, 3.15, 0.2)
        linear_model = (np.mean(samples['a'])*thetas) + np.mean(samples['b'])

        ax.plot(thetas, linear_model, c=red, ls='-',
                label=f"$a$ = {np.mean(samples['a']):.3f} $\\pm$ {np.std(samples['a']):.3f}\n$b$ = {np.mean(samples['b']):.3f} $\\pm$ {np.std(samples['b']):.3f}")
        ax.fill_between(thetas, linear_model-np.mean(np.sqrt(samples['sigma_2_res'])), linear_model+np.mean(np.sqrt(samples['sigma_2_res'])), color=red, alpha=0.2,
                        label="$\\sigma_{\\rm{res}, \\theta}$ ="+f"{np.mean(np.sqrt(samples['sigma_2_res'])):.3f} $\\pm$ {np.std(np.sqrt(samples['sigma_2_res'])):.3f}")

    ax.legend()
    ax.set_xlabel('$\\theta_{gri}$')
    ax.set_ylabel('$M_{z, \\rm{int}}$')

    fig.align_labels()
    fig.tight_layout()
    return fig, ax



def compute_mass_step(x, y, yerr, M_split=10, survey_mask=None, hmc_kwargs={}):

    def model(x, y, yerr, M_split, survey_mask):
        mass_mask = x < M_split

        delta_LM = numpyro.sample("delta_LM", dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
        gamma = numpyro.sample("gamma", dist.ImproperUniform(dist.constraints.real, (), event_shape=()))

        if survey_mask is None:
            sigma_2_res_LM = numpyro.sample("sigma_2_res_LM", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_res_HM = numpyro.sample("sigma_2_res_HM", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))

        else:
            sigma_2_res_LM_YSE = numpyro.sample("sigma_2_res_LM_YSE", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_res_HM_YSE = numpyro.sample("sigma_2_res_HM_YSE", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_res_LM_Foundation = numpyro.sample("sigma_2_res_LM_Foundation", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            sigma_2_res_HM_Foundation = numpyro.sample("sigma_2_res_HM_Foundation", dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))

            sigma_2_res_LM = sigma_2_res_LM_YSE*survey_mask + sigma_2_res_LM_Foundation*(1-survey_mask)
            sigma_2_res_HM = sigma_2_res_HM_YSE*survey_mask + sigma_2_res_HM_Foundation*(1-survey_mask)

        numpyro.sample("obs", dist.Normal(delta_LM + gamma*(1-mass_mask), jnp.sqrt(sigma_2_res_LM*mass_mask + sigma_2_res_HM*(1-mass_mask) + yerr**2)), obs=y)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    
    kernel = NUTS(model)
    num_warmup = hmc_kwargs.pop('num_warmup', 5000)
    num_samples = hmc_kwargs.pop('num_samples', 15_000)
    sampler = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, **hmc_kwargs)
    sampler.run(rng_key_, x, y, yerr, M_split, survey_mask)
    
    return sampler



def plot_mass_step(data, fit_filters='griz', M_split=10, samples=None, dust_corrected=True):

    fig, ax = plt.subplots(1, 1, figsize=(8,4), sharex=True)
    ax.axvline(M_split, color='dimgray')

    survey_mask = data['SURVEY'] == 'YSE'
    redshift_final, mu_LCDM = model_for_plotting(data['REDSHIFT_FINAL'])
    sigma = get_peculiar_velocity_unc(redshift_final, sigma_z=data['SIGMA_Z'])
    x = data['LOG_HOSTMASS']
    y = data[f'{fit_filters.upper()}_MU'] - mu_LCDM
    yerr = np.sqrt(data[f'{fit_filters.upper()}_MU_ERR']**2 + sigma**2)

    if dust_corrected and len(fit_filters) == 1:
        wave = wave_dict[fit_filters]
        ext_mag = np.zeros(len(data))
        for r, row in enumerate(data):
            ext_mag[r] = extinction.fitzpatrick99(np.array([wave]), a_v = row['GRI_AV'], r_v=2.610)

        y = y - ext_mag

    _, _, bars = ax.errorbar(x[~survey_mask], y[~survey_mask], yerr[~survey_mask], color=blue, ls='None', marker='o', ms=5)
    [bar.set_alpha(0.25) for bar in bars]
    _, _, bars = ax.errorbar(x[survey_mask], y[survey_mask], yerr[survey_mask], color=green, ls='None', marker='s', ms=5)
    [bar.set_alpha(0.25) for bar in bars]

    if not samples == None:
        xmin, xmax = ax.get_xlim()
        xwidth = xmax - xmin
        ymin, ymax = ax.get_ylim()
        ywidth = ymax - ymin

        delta_LM = np.mean(samples['delta_LM'])
        delta_LM_unc = np.std(samples['delta_LM'])
        delta_HM = np.mean(samples['delta_LM']) + np.mean(samples['gamma'])
        delta_HM_unc = np.sqrt(np.std(samples['gamma'])**2 - np.std(samples['delta_LM'])**2)

        ax.axhline(delta_LM, 0, (M_split - xmin)/xwidth, ls='-', color='crimson')
        ax.axhspan(delta_LM-delta_LM_unc, delta_LM+delta_LM_unc, 0, (M_split - xmin)/xwidth, ls='-', color='crimson', alpha=0.2)

        ax.axhline(delta_HM, (M_split - xmin)/xwidth, 1, ls='-', color='crimson')
        ax.axhspan(delta_HM-delta_HM_unc, delta_HM+delta_HM_unc, (M_split - xmin)/xwidth, 1, ls='-', color='crimson', alpha=0.2)

        try:
            ax.text(6.75, ymin+(0.24*ywidth), '$\\sigma_{\\rm{res, LM}}$'+f": {np.mean(np.sqrt(samples['sigma_2_res_LM'])):.3f} $\\pm$ {np.std(np.sqrt(samples['sigma_2_res_LM'])):.3f}", fontsize=16)
            ax.text(6.75, ymin+(0.1*ywidth), '$\\sigma_{\\rm{res, HM}}$'+f": {np.mean(np.sqrt(samples['sigma_2_res_HM'])):.3f} $\\pm$ {np.std(np.sqrt(samples['sigma_2_res_HM'])):.3f}", fontsize=16)
        except:
            ax.text(6.75, ymin+(0.3*ywidth), '$\\sigma_{\\rm{res, LM}}$: ', fontsize=16)
            ax.text(7.5, ymin+(0.3*ywidth), f"{np.mean(np.sqrt(samples['sigma_2_res_LM_Foundation'])):.3f} $\\pm$ {np.std(np.sqrt(samples['sigma_2_res_LM_Foundation'])):.3f}", fontsize=16, color=blue)
            ax.text(7.5, ymin+(0.22*ywidth), f"{np.mean(np.sqrt(samples['sigma_2_res_LM_YSE'])):.3f} $\\pm$ {np.std(np.sqrt(samples['sigma_2_res_LM_YSE'])):.3f}", fontsize=16, color=green)
            
            ax.text(6.75, ymin+(0.12*ywidth), '$\\sigma_{\\rm{res, HM}}$: ', fontsize=16)
            ax.text(7.5, ymin+(0.12*ywidth), f"{np.mean(np.sqrt(samples['sigma_2_res_HM_Foundation'])):.3f} $\\pm$ {np.std(np.sqrt(samples['sigma_2_res_HM_Foundation'])):.3f}", fontsize=16, color=blue)
            ax.text(7.5, ymin+(0.04*ywidth), f"{np.mean(np.sqrt(samples['sigma_2_res_HM_YSE'])):.3f} $\\pm$ {np.std(np.sqrt(samples['sigma_2_res_HM_YSE'])):.3f}", fontsize=16, color=green)

        rms = np.sqrt(np.mean(np.array(y)**2))
        ax.text(6.75, ymin+(0.85*ywidth), f"$\\gamma$: {np.mean(samples['gamma']):.3f} $\\pm$ {np.std(samples['gamma']):.3f}", fontsize=16)

    ax.set_xlabel('$\\log_{10}(M_{*} / M_{\odot})$')
    ax.set_ylabel('$\\mu_{'+fit_filters+'} - \\mu_{\\Lambda \\rm{CDM}}(z)$')

    return fig, ax