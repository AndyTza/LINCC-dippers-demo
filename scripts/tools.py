"""
Tools for analysis.
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.modeling.models import Gaussian1D
import pandas as pd
from astropy.modeling import fitting
import heapq
from astroquery.gaia import Gaia
from statsmodels.tsa import stattools
from scipy import stats
from stetson import stetson_j, stetson_k
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

bad_times = np.zeros(100)

def expandable_window(xdat, ydat, cent, atol=0.001):
    """ Calcualte the window size for a given peak. 
    
    Parameters:
    -----------
    xdat (array-like): Input time values.
    ydat (array-like): Input deviation values.
    cent (float): Center of the peak.

    Returns:
    --------
    window_end_pos (float): Window end for the positive side.
    window_end_neg (float): Window end for the negative side.
    """
    phase = xdat - cent
    
    pos_condition = phase > 0
    neg_condition = phase < 0

    p_x_pos, p_y_pos = xdat[pos_condition], ydat[pos_condition]
    p_xs_pos = np.argsort(p_x_pos)[::-1]

    p_x_neg, p_y_neg = xdat[neg_condition], ydat[neg_condition]
    p_xs_neg = np.argsort(p_x_neg)

    window_end_pos = find_window_end(p_x_pos[p_xs_pos], p_y_pos[p_xs_pos], atol=atol)
    window_end_neg = find_window_end(p_x_neg[p_xs_neg], p_y_neg[p_xs_neg], atol=atol)

    return window_end_pos, window_end_neg

def find_window_end(p_x, p_y, dif=1.86, atol=0.005):
    """ Find the window end for a given peak by searching the neighbouring until the difference is minimized.
    
    Parameters:
    -----------
    p_x (array-like): Input time values.
    p_y (array-like): Input deviation values.

    Returns:
    --------
    window_end (float): Window end.
    """
    window_end = 0
    for i in range(len(p_x) - 1):
        xi, yi = p_x[i], p_y[i]
        xi2, yi2 = p_x[i + 1], p_y[i + 1]

        delta_diff = yi2 - yi

        if np.isclose(delta_diff, dif, atol=atol):
            window_end = xi2

    return window_end
    

def prepare_lc(time, mag, mag_err, flag, band, band_of_study='r', flag_good=0, q=None, custom_q=False, bad_times=bad_times):
    """
    Prepare the light curve for analysis - specifically for the ZTF data.
    
    Parameters:
    -----------
    time (array-like): Input time values.
    mag (array-like): Input magnitude values.
    mag_err (array-like): Input magnitude error values.
    flag (array-like): Input flag values.
    band (array-like): Input band values.
    band_of_study (str): Band to study. Default is 'r' band
    flag_good (int): Flag value for good detections. Default is 0 (see ZTF documentation)

    Returns:
    --------
    time (array-like): Output time values.
    mag (array-like): Output magnitude values.
    mag_err (array-like): Output magnitude error values.
    """
    try:
        if len(time)==0:
            return np.array([0]), np.array([0]), np.array([0])
        if custom_q:
            rmv = q
        else:
            # Selection and preparation of the light curve (default selection on )
            rmv = (flag == flag_good) & (mag_err>0) & (band==band_of_study) & (~np.isnan(time)) & (~np.isnan(mag)) & (~np.isnan(mag_err)) # remove nans!
        
        time, mag, mag_err = time[rmv], mag[rmv], mag_err[rmv]
        #plt.figure(figsize=(15,5))
        #plt.scatter(time, mag, color='gray')

        # Sort times for later use...
        srt = time.argsort()
        time, mag, mag_err = time[srt], mag[srt], mag_err[srt]

        #plt.figure()
        #if band_of_study == 'r':
        #    plt.scatter(time, mag, color='gray', s=90)

        # let's apply the old code to our example
        # Remove observations that are >0.9 days apart
        #dts = np.diff(time)
        #_id = dts[np.argsort(dts)]>=0.9
        #time, mag, mag_err = time[np.argsort(dts)][_id], mag[np.argsort(dts)][_id], mag_err[np.argsort(dts)][_id] # sort the values
        time, mag, mag_err = time.to_numpy(), mag.to_numpy(), mag_err.to_numpy()
        # Remove observations that are >0.9 days apart without sorting
        dts = np.diff(time)
        mask = dts >= 0.9
        # Apply the mask to get the indices of valid points
        # Note that `dts` has one less element than `time`, so we need to handle indexing carefully
        filtered_indices = np.insert(mask, 0, True)  # Keep the first point by default

        # Filter `time`, `mag`, and `mag_err`
        time, mag, mag_err = time[filtered_indices], mag[filtered_indices], mag_err[filtered_indices]
        #plt.scatter(time, mag, color='crimson', marker='x')

        # make them into numpy arrys
        #if band_of_study == 'r':
        #    plt.scatter(time, mag, color='k', s=30)

        # Sort times for later use AGAIN....
        srt = time.argsort()
        time, mag, mag_err = time[srt], mag[srt], mag_err[srt]

        #if band_of_study == 'r':
        #    plt.scatter(time, mag, color='blue', s=10)

        #if band_of_study == 'g':
        #   plt.scatter(time, mag, s=20, color='k')


        # TODO: old cadence remover...
        #TLDR: this was not the best method for removing high cadence fields. Many systematics were still passing...
        #cut_close_time = np.where(np.diff(time) < 1)[0] + 1
        #time, mag, mag_err  = np.delete(time, cut_close_time), np.delete(mag, cut_close_time), np.delete(mag_err, cut_close_time)

        
        # Remove observations that are <0.5 day apart
        #cut_close_time = np.where(np.diff(time) <1)[0] - 1
        #time, mag, mag_err  = np.delete(time, cut_close_time), np.delete(mag, cut_close_time), np.delete(mag_err, cut_close_time)

        #cut_close_time = np.where(np.diff(time) <0.5)[0] + 1
        #time, mag, mag_err  = np.delete(time, cut_close_time), np.delete(mag, cut_close_time), np.delete(mag_err, cut_close_time)

        return time, mag, mag_err
    except:
        return np.array([0]), np.array([0]), np.array([0])


def fill_gaps(time, mag, mag_err, max_gap_days=90, num_points=20, window_size=0):
    """Fill the seasonal gaps of my data with synthetic observations based on the previous detections.
    
    Parameters:
    -----------
    time (array-like or pandas.core.series.Series): Input time values.
    mag (array-like or pandas.core.series.Series): Input magnitude values.
    mag_err (array-like or pandas.core.series.Series): Input magnitude error values.
    max_gap_days (float): Maximum gap size in days. Default is 90 days.
    num_points (int): Number of synthetic points to generate. Default is 20.
    window_size (int): Number of previous detections to use for the mean and standard deviation. Default is 10.

    Returns:
    --------
    filled_time (array-like or pandas.core.series.Series): Output time values.
    filled_mag (array-like or pandas.core.series.Series): Output magnitude values.
    filled_mag_err (array-like or pandas.core.series.Series): Output magnitude error values.

    """
    
    # Identify the indices where there are gaps greater than max_gap_days
    dts = np.diff(time)
    where_big = np.where(dts > max_gap_days)[0]
    
    # Initialize arrays to store filled data
    filled_time, filled_mag, filled_mag_err = time.copy(), mag.copy(), mag_err.copy()

    for i in where_big:
        # Determine the start and end indices of the gap
        start_idx = i
        end_idx = i + 1

        # Calculate median from the last 5 and next 5 detections
        last5_median = np.nanmedian(mag[max(0, start_idx - 5):start_idx])
        next5_median = np.nanmedian(mag[end_idx:end_idx + 5])
        mean_mag = np.linspace(last5_median, next5_median, num_points)
        
        # Calculate standard deviation from the last 5 and next 5 detections
        last5_std = np.nanstd(mag[max(0, start_idx - 5):start_idx])
        next5_std = np.nanstd(mag[end_idx:end_idx + 5])
        std_mag = np.linspace(last5_std, next5_std, num_points)

        # Generate synthetic points within the gap
        synthetic_time = np.linspace(time[start_idx]+10, time[end_idx], num_points)
        synthetic_mag = np.random.normal(loc=mean_mag, scale=std_mag)
        synthetic_mag_err = np.random.normal(loc=np.nanmean(mag_err), scale=np.nanstd(mag_err), size=num_points)

        # Add additional modification without overlapping old data
        mask = (synthetic_time >= time[start_idx]) & (synthetic_time <= time[end_idx])
        filled_time = np.concatenate([filled_time[:end_idx], synthetic_time[mask], filled_time[end_idx:]])
        filled_mag = np.concatenate([filled_mag[:end_idx], synthetic_mag[mask] + np.random.normal(0, 0.2 * np.nanstd(mag), np.sum(mask)), filled_mag[end_idx:]])
        filled_mag_err = np.concatenate([filled_mag_err[:end_idx], synthetic_mag_err[mask] + np.random.normal(0, 1*np.nanstd(mag_err), np.sum(mask)), filled_mag_err[end_idx:]])

    # Sort the filled data by time
    xs = np.argsort(filled_time)
    
    # Convert the filled data back to pandas Series if the input was pandas Series
    if isinstance(time, pd.core.series.Series):
        filled_time = pd.Series(filled_time[xs])
    if isinstance(mag, pd.core.series.Series):
        filled_mag = pd.Series(filled_mag[xs])
    if isinstance(mag_err, pd.core.series.Series):
        filled_mag_err = pd.Series(filled_mag_err[xs])
    
    return filled_time, filled_mag, filled_mag_err


def digest_the_peak(peak_dict, time, mag, mag_err, expandby=0):
    """Given the peak dictionary and data - prepare my light curve for GP analysis and integration.
    
    Parameters:
    -----------
    peak_dict (dict): Dictionary of the peak.
    time (array-like): Input time values.
    mag (array-like): Input magnitude values.
    mag_err (array-like): Input magnitude error values.
    expandby (float): Number of days to expand the window by. Default is 0 days.

    Returns:
    --------
    time (array-like): Output time values.
    mag (array-like): Output magnitude values.
    mag_err (array-like): Output magnitude error values.
    """
    # Define starting pontnts
    # TODO: make sure correct order
    try:
        Ns = len(time)
    except:
        return np.array([0]), np.array([0]), np.array([0])

    try:
        start, end = peak_dict['window_start'].values[0], peak_dict['window_end'].values[0]
    except:
        try:
            Ns = len(time)
            start, end = min(time), max(time)
        except:
            return np.array([0]), np.array([0]), np.array([0])
    
    if (start==0) or (end==0):
        return np.array([0]), np.array([0]), np.array([0])
    else:
        selection = np.where((time > start-expandby) & (time < end+expandby) & (~np.isnan(time)) & (~np.isnan(mag)) & (~np.isnan(mag_err)))

        return time[selection], mag[selection], mag_err[selection]

def estimate_gaiadr3_density_async(ra_target, dec_target, radius=0.01667):
    """Do this assync to avoid timeouts."""
    query1 = f"""SELECT
                ra, dec, parallax, phot_g_mean_mag,
                DISTANCE(
                    POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
                    POINT('ICRS',{ra_target}, {dec_target})
                ) AS separation
                FROM gaiadr3.gaia_source
                WHERE 1=CONTAINS(POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
                                CIRCLE('ICRS',{ra_target}, {dec_target},1./60.))
                ORDER BY separation ASC
                            """
    tbl = Gaia.launch_job(query1, dump_to_file=False).get_results()

    return {"closest_bright_star_arcsec": tbl[np.argmin(tbl['phot_g_mean_mag'])]['separation'],
    "closest_bright_star_mag": tbl[np.argmin(tbl['phot_g_mean_mag'])]['phot_g_mean_mag'], 
    "closest_star_arcsec": tbl['separation'][0],
    "closest_star_mag": tbl['phot_g_mean_mag'][0],
    "density_arcsec2": len(tbl)/(np.pi*radius**2)}


def estimate_gaiadr3_density(ra_target, dec_target, gaia_lite_table, radius=0.01667):
    """
    Estimate the density of stars in the Gaia DR3 catalog around a given target.

    Parameters:
    -----------
    ra_target (float): Right ascension of the target in degrees.
    dec_target (float): Declination of the target in degrees.
    radius (float): Radius of the cone search in degrees. Default is 1 arcmin (1arcmin ~0.01667 deg).
    gaia_lite_table (LSDB hipscat table): Gaia DR3 table. Default is the one loaded in the notebook.

    Returns:
    --------
    closest_star_arcsec (float): Separation in arcseconds of the closest star.
    closest_star_mag (float): Magnitude of the closest star.
    density_arcsec2 (float): Density of stars in arcsec^-2.
    """

    sky_table = gaia_lite_table.cone_search(ra=ra_target, dec=dec_target, radius=radius).compute()
    
    if len(sky_table) == 0:
        {"closest_bright_star_arcsec": 0,
        "closest_bright_star_mag": 0, 
        "closest_star_arcsec": 0,
        "closest_star_mag": 0,
        "density_arcsec2": 0}

    sky = SkyCoord(ra=sky_table['ra'].values*u.deg, dec=sky_table['dec'].values*u.deg, frame='icrs')
    sky_target = SkyCoord(ra=ra_target*u.deg, dec=dec_target*u.deg, frame='icrs') # sky coord of object of interest

    # Find sky separation in arcseconds
    delta_sep = sky.separation(sky_target).to(u.arcsec).value

    # Find the separation to the cloest star.
    sep_sorted = np.argsort(delta_sep) 

    return {"closest_bright_star_arcsec": delta_sep[np.argmax(sky_table['phot_g_mean_mag'].values)]/60/60,
        "closest_bright_star_mag": sky_table['phot_g_mean_mag'].values[np.argmin(sky_table['phot_g_mean_mag'].values)], 
        "closest_star_arcsec": delta_sep[sep_sorted][0]/60/60,
        "closest_star_mag": sky_table['phot_g_mean_mag'].values[sep_sorted][0],
        "density_arcsec2": len(delta_sep)/np.pi/radius**2}


def bin_counter(xdat, ydat, bin_width=3):
    """
    Calculate the running median.

    Parameters:
    -----------
    xdat (array-like): Array of x-values.
    ydat (array-like): Array of y-values.
    bin_width (float): Width of each bin. Default is 3.

    Returns:
    --------
    bin_centroid (ndarray): Array of bin centroids.
    running_median (ndarray): Array of running medians of y-values.
    """

    if len(xdat) == 0:
        return [None], [None]

    # Calculate the number of bins
    Nbins = int((max(xdat) - min(xdat)) / bin_width)

    # Initialize arrays to store the bin centroids and running medians
    bin_centroid = np.zeros(Nbins)
    running_median = np.zeros(Nbins)

    # Calculate the running median
    for i in range(Nbins):
        # Define the bin edges
        bin_start = min(xdat) + i * bin_width
        bin_end = bin_start + bin_width

        # Select the data points within the bin
        bin_data = ydat[(xdat >= bin_start) & (xdat < bin_end)]

        # Calculate the bin centroid
        bin_centroid[i] = bin_start + 0.5 * bin_width

        # Calculate the running median
        running_median[i] = np.nanmedian(bin_data)
    
    return bin_centroid, running_median


def quick_Gaussian_fit(time, running_deviation):
    """LevMarSQFitter Gaussian fit to the running deviation.
    
    Parameters:
    -----------
    time (array-like): Array of time values.
    running_deviation (array-like): Array of running deviation values.

    Returns:
    --------
    summary (dict): Dictionary of the Gaussian fit parameters.
    """
    t_init = Gaussian1D(amplitude=50., mean=time[np.argmax(running_deviation)]-1, stddev=1, 
                          bounds={"amplitude": (3, max(running_deviation)), 
                                 "mean": (time[np.argmax(running_deviation)]-535,
                                          time[np.argmax(running_deviation)]+535), 
                                 "stddev": (0.1, 1_000)})
    fit_t = fitting.LevMarLSQFitter()
    t = fit_t(t_init, time, running_deviation, maxiter=2_000)
    return dict(amplitude=t.amplitude.value, mean=t.mean.value, stddev=t.stddev.value)


def calc_Stetson(mag, err, N, wmean):
    """Calculate the Welch/Stetson I and Stetson J,K statistics
    e.g. https://www.aanda.org/articles/aa/pdf/2016/02/aa26733-15.pdf.
    Source: https://github.com/ZwickyTransientFacility/scope/blob/5a67134ab2cf41d0aab2c1b8920f63a67b48362e/tools/featureGeneration/lcstats.py#L69
    """
    d = np.sqrt(1.0 * N / (N - 1)) * (mag - wmean) / err
    P = d[:-1] * d[1:]

    # stetsonI
    stetsonI = np.sum(P)

    # stetsonJ
    stetsonJ = np.sum(np.sign(P) * np.sqrt(np.abs(P)))

    # stetsonK
    stetsonK = np.sum(abs(d)) / N
    stetsonK /= np.sqrt(1.0 / N * np.sum(d**2))

    return stetsonI, stetsonJ, stetsonK


def calc_invNeumann(t, mag, wstd):
    """Calculate the time-weighted inverse Von Neumann stat.
        Source: https://github.com/ZwickyTransientFacility/scope/blob/5a67134ab2cf41d0aab2c1b8920f63a67b48362e/tools/featureGeneration/lcstats.py#L69
    """
    dt = t[1:] - t[:-1]
    dm = mag[1:] - mag[:-1]

    w = (dt) ** -2  # inverse deltat weighted
    eta = np.sum(w * dm**2)
    eta /= np.sum(w) * wstd**2

    return eta**-1


def calc_NormExcessVar(mag, err, N, wmean):
    """    Source: https://github.com/ZwickyTransientFacility/scope/blob/5a67134ab2cf41d0aab2c1b8920f63a67b48362e/tools/featureGeneration/lcstats.py#L69
    """
    stat = np.sum((mag - wmean) ** 2 - err**2)
    stat /= N * wmean**2
    return stat


def other_summary_stats(x, y, yerr, N, wmean, wstd): 
    """Calculation of other random time series statistics.
       Here is a list of all our evaluated features: 
       - Skew
       - Kurtosis
       - Stetson J
       - Stetson K

    Parameters:
    -----------
    y (array-like): Array of magnitudes.
    """

    try:
        return {"skew": stats.skew(y),
            "kurtosis": stats.kurtosis(y),
            "stetson_I": calc_Stetson(y, yerr, N, wmean)[0],
            "stetson_J": calc_Stetson(y, yerr, N, wmean)[1],
            "stetson_K": calc_Stetson(y, yerr, N, wmean)[2],
            "invNeumann": calc_invNeumann(x, y, wstd)}
    except:
        return {"skew": np.nan,
            "kurtosis": np.nan, 
            "stetson_I": np.nan,
            "stetson_J": np.nan,
            "stetson_K": np.nan,
            "invNeumann": np.nan}


def chidof(y):
    """Calculate the chi-square dof under a constant model.
    
    Parameters:
    -----------
    y (array-like): Array of magnitudes.
    """

    if len(y)==0:
        return np.nan
    else:
        Ndata = len(y)
        M, S = np.nanmedian(y), np.nanstd(y)
        return 1/Ndata * sum(((y-M)/(S))**2)

def adf_tests(y):
    """Perform the ADF test on the light curve.
    
    Parameters:
    -----------
    y (array-like): Array of magnitudes.

    Returns:
    --------
    Dictionary summarizing: 
        ADF-const (float): ADF test statistic for constant model.
        p-const (float): p-value for constant model.
        ADF-const-trend (float): ADF test statistic for constant + trend model.
        p-const-trend (float): p-value for constant + trend model.
    """

    if len(y) < 10 : # TODO: what is the minimum number of points to perform the test? Requires at least 10 detections for a regression...
        return {"ADF-const": np.nan, "p-const":np.nan,
                "ADF-const-trend": np.nan, "p-const-trend":np.nan}

    else:

        try: # sadly ADF sometimes won't work due to issues with LinearRegression fit.? #TODO: investigate
            # Constant 
            stat_1 = stattools.adfuller(y,
                        maxlag=None, 
                        regression='c', 
                        autolag='AIC', 
                        store=False,
                        regresults=False)

            # Constant + Trend
            stat_2 = stattools.adfuller(y,
                        maxlag=None, 
                        regression='ct', 
                        autolag='AIC', 
                        store=False,
                        regresults=False)

            return {"ADF-const": stat_1[0], "p-const":stat_1[1], 
                    "ADF-const-trend": stat_2[0], "p-const-trend":stat_2[1]}
        except:
            return {"ADF-const": np.nan, "p-const":np.nan,
                    "ADF-const-trend": np.nan, "p-const-trend":np.nan}


def gaus(x, a, x0, sigma, ofs):
    """"Calculate a simple Gaussian function with a term offset"""
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + ofs

def auto_fit(x, y, yerr, loc, base, return_model=False):
    """Perform a Gaussian function auto fitting with some lose priors."""
    try:
        popt, pcov = curve_fit(gaus, 
                                x,
                                y,
                                p0=[1, loc, 7, base],
                            bounds=((0.1, loc-5, 0.1, base-2),
                                    (np.inf, loc+5, np.inf, base+2)))
    except: # if fails return zeros...
        popt, pcov = [0, 0, 0, 0], [0, 0, 0, 0]
    
    if return_model:
        model = gaus(x, *popt)
        chi = (y - model) / yerr
        return model, np.sum(chi**2)
    else:
        #popt[2] += 50
        model = gaus(x, *popt)
        chi = (y - model) / yerr
        #plt.figure(figsize=(25,5))
        #plt.scatter(x, y)
        #plt.plot(x, model)
        #plt.axvline(loc)
        #plt.xlim(loc-650, loc+650)
        return popt, np.sum(chi**2)

def fwhm_calc(pop):
    """Calculate the FWFM of the Gaussian fit.
        Parameters:
        -----------
        pop (array-like): Array of Gaussian fit parameters.
    """
    return 2.355 * pop[2] # fwhm 

def calc_sum_score(xdat, ydat, yerr, peak_dict, base, rms, out_g):
    """Calculate the light curve score."""
    
    score_term = 0
    for i in range(peak_dict[0]):
        event = peak_dict[1][f'dip_{i}']

        loc = event['peak_loc']
        powr = event['dip_power']
        Ndet = event['N_1sig_in_dip']
        ts, te = event['window_start']+1, event['window_end']-1
        
        eval0 = auto_fit(xdat, ydat, yerr, loc, base, return_model=False)
        fit_temrs, chi2 = eval0[0], eval0[1]
        
        
        ww = np.where((xdat > ts - 2*fit_temrs[2]) & (xdat < te + 2*fit_temrs[2]))
        xx, yy, yerrr = xdat[ww], ydat[ww], yerr[ww]
        
        ww2 = np.where((xdat > ts - fit_temrs[2]) & (xdat < te + fit_temrs[2]))
        xx2, yy2, yerrr2 = xdat[ww2], ydat[ww2], yerr[ww2]
        
        # XTRA WIDE
        wwW = np.where((xdat > ts - 10*fit_temrs[2]) & (xdat < te + 10*fit_temrs[2]))
        xxW, yyW, yerrrW = xdat[wwW], ydat[wwW], yerr[wwW]
        #plt.scatter(xx2, yy2, color='k')

        model_trim = gaus(xx, *fit_temrs)
        pad = 0.03
        MAX_MODEL = max(model_trim) - pad
          
        #plt.axhline(MAX_MODEL)
        #plt.axhline(np.mean(ydat), color='red')
        #print(abs(MAX_MODEL-np.mean(ydat)))
        #print (MAX_MODEL, np.std(ydat), np.median(ydat), abs(np.median(ydat)-MAX_MODEL))
        
        residual = (yy - model_trim) 
        
        chi_residual = np.sum(residual**2)
        fwhm = fwhm_calc(fit_temrs)
        #print (fwhm, chi_residual)
        
        if abs(MAX_MODEL-np.mean(ydat))<=0.05 and peak_dict[0]==1:
            # impose 5sigma cut
            if MAX_MODEL>np.mean(ydat)-5*np.std(ydat) and MAX_MODEL<np.mean(ydat)+5*np.std(ydat):
                #print (f"ALERT.........................{abs(MAX_MODEL-np.mean(ydat))}...................signal is too weak")
                fwhm = -99
                
        #print (abs(MAX_MODEL-np.mean(ydat)), fwhm, 'og')
        
        if abs(MAX_MODEL-np.mean(ydat))<0.08 and fwhm<3:
            fwhm = -99 
            score_term += -8_000
            
        
        if fwhm<=4:
            expand_factor = 2
        else:
            expand_factor = 8
        qqloc = np.where((xdat>loc-expand_factor) & (xdat<loc+expand_factor))
        
        xx3, yy3, yerrr3 = xdat[qqloc], ydat[qqloc], yerr[qqloc]
        #plt.scatter(xx3, yy3, color='purple', s=50, alpha=1)
        
        Nwithin_five = len(xdat[qqloc]) # number of detections +/5 days within loc

        qqloc2 = np.where((xdat>loc-2) & (xdat<loc+2))
        Nwithin_two = len(xdat[qqloc2]) # number of detections +/2 days within loc
        
        

        #print (chi_residual)
        
        #print (Nwithin_five, Nwithin_two)
        #print (chi_residual)
        #print ('---<', out_g)
        # TODO: the challenge here is that we're trying to figure out of the FWHM selected is appropriate from the model....
        #print (Nwithin_five)
        if fwhm/len(xx2) < 0.1 or Nwithin_five<=1:
            # if the residual is within this bound (aka likely too good to be true...)
            if chi_residual<2e-3:
                # penalty on FWHM
                fwhm = 0.0001
            else:
                fwhm = fwhm
        
        if chi_residual<1e-3:
            # If the residual is very small let's investigate and see if it's a legit source
            N_ratios = Nwithin_two/Nwithin_five
            y_at_peak = np.nanmean(ydat[qqloc])
            #print ('------>', N_ratios * out_g)
            #plt.axhline(y_at_peak, lw=5, color='r', alpha=0.2)
            #plt.axvline(loc-7)
            #plt.axvline(loc+7)

            # how many are above the peak plus some wiggle room
            above_peak = np.where((ydat[qqloc] > y_at_peak))[0]

            N_above_peak = len((ydat[qqloc])[above_peak])

            # how many are below the peak plus some wiggle room
            #print (N_above_peak)

            if N_above_peak>2:
                chi_residual=chi_residual
            else:
                # penalty on chi_residual (i.e., cannot have perfect fits...)
                # inflate because it means you're fitting a very small number of points...
                chi_residual = 1e5
                # Ndet or len(xx2)
        #print ('COUNTER:',Ndet, len(xx2), Nwithin_two)
        #print ("MIN-MAX of FHWM:", min(xx2), max(xx2))
        #print (fwhm, peak_dict[0])
        if fwhm<1.45 and peak_dict[0]>=1: # TODO NEW FEATURE
            score_term += 0
        else:
            score_term += fwhm * (powr/2) * (Ndet) * 1/(chi_residual)
 
    return  1/(peak_dict[0] * np.log(peak_dict[0]+1)) * (1/1) * score_term, min(xx3), max(xx3), abs(MAX_MODEL-np.mean(ydat))