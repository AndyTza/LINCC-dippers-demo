import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings('ignore')

#Gaussian process modules
from george import kernels
import george
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared 
from scipy import interpolate
import astropy.stats as astro_stats
from scipy.optimize import curve_fit
from astropy.io import ascii
from scipy.signal import find_peaks
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt
from tools import expandable_window


_all_funcs = ["deviation", 
                "calc_dip_edges", 
                "GaussianProcess_dip", 
                "calculate_integral",
                 "calculate_assymetry_score", 
                 "evaluate_dip", 
                 "light_curve_ens"]


def deviation(mag, mag_err, R, S):
    """Calculate the running deviation of a light curve for outburst or dip detection.
    
    d >> 0 will be dimming
    d << 0 (or negative) will be brightenning
    
    
    Parameters:
    -----------
    mag (array-like): Magnitude values of the light curve.
    mag_err (array-like): Magnitude errors of the light curve.
    R (float): Biweight location of the light curve (global).
    S (float): Biweight scale of the light curve (global).

    Returns:
    --------
    dev (array-like): Deviation values of the light curve.
    """
    # Calculate biweight estimators
    return (mag - R) / np.sqrt(mag_err**2 + S**2)  


def detect_bursts_edges(time, mag, center_time, baseline_mean, baseline_std, burst_threshold=3.0, expansion_indices=1):
    """
    Detect bursts in a time series using linear interpolation.

    Parameters:
    -----------
    time (array-like): Time values of the light curve.
    mag (array-like): Magnitude values of the light curve.
    center_time (float): Center time of the burst.
    baseline_mean (float): Mean of the baseline.
    baseline_std (float): Standard deviation of the baseline.
    burst_threshold (float): Threshold for burst detection. Default is 3.0.
    expansion_indices (int): Number of indices to expand the burst region. Default is 1.

    Returns:
    --------
    burst_start (float): Start time of the burst.
    burst_end (float): End time of the burst.
    """

    # Define a linear interpolation function
    interpolate_flux = np.interp

    # Initialize burst_start and burst_end
    burst_start = burst_end = np.searchsorted(time, center_time)

    # Find burst start
    while burst_start > 0:
        burst_start -= 1
        if mag[burst_start] < baseline_mean + burst_threshold * baseline_std:
            break

    # Find burst end
    while burst_end < len(time) - 1:
        burst_end += 1
        if mag[burst_end] < baseline_mean + burst_threshold * baseline_std:
            break

    # Expand burst region towards the beginning
    burst_start = max(0, burst_start - expansion_indices)

    # Expand burst region towards the end
    burst_end = min(len(time) - 1, burst_end + expansion_indices)

    # Final start and end
    t_start, t_end = time[burst_start], time[burst_end]

    # How many detections above 2std above the mean?
    N_thresh_1 = len((mag[(time>t_start) & (time<t_end)]>baseline_mean + 2*baseline_std))

    return t_start, t_end, abs(t_start-center_time), abs(t_end-center_time), N_thresh_1, 0, 0

def calc_dip_edges(xx, yy, _cent, atol=1e-32):
    """ Calculate the edges of a dip given the center dip time. 

    Parameters:
    -----------
    xx (array-like): Time values of the light curve.
    yy (array-like): Magnitude values of the light curve.
    _cent (float): Center time of the dip.
    atol (float): Tolerance for the edge calculation. Default is 0.01.

    Returns:
    --------
    t_forward (float): Forward edge of the dip.
    t_back (float): Backward edge of the dip.
    time forward difference (float): Time difference between the forward edge and the center.
    time backward difference (float): Time difference between the backward edge and the center.
    N_thresh_1 (int): Number of detections above the median threshold in the given window.
    t_in_window (float): Average time difference in the given window.
    """
    
    indices_forward = np.where((xx > _cent) & np.isclose(yy, np.nanmean(yy) - 0.5*np.nanstd(yy), atol=atol))[0]
    t_forward = xx[indices_forward[0]] if indices_forward.size > 0 else 0
    
    # select indicies close to the center (negative)
    indices_back = np.where((xx < _cent) & np.isclose(yy, np.nanmean(yy) - 0.5*np.nanstd(yy), atol=atol))[0]
    if indices_back.size > 0:
        t_back = xx[indices_back[-1]]
    else:
        t_back = 0

    #TODO: might require a more expantable version...
    #TODO: impose requirement to have 5 detections on the baseline of the dip
    #TODO: give me the first 5 detections that are within the quntiles of the data...

    #t_forward, t_back = expandable_window(xx, yy, _cent, atol=atol)

    # Diagnostics numbers
    # How many detections above the median thresh in the given window?
    _window_ = (xx>t_back) & (xx<t_forward)

    sel_1_sig = (yy[_window_]>np.nanmedian(yy) + 1*np.nanstd(yy)) # detections above 1 sigma
    N_thresh_1 = len((yy[_window_])[sel_1_sig])
    N_in_dip = len((yy[_window_]))

    # select times inside window and compute the average distance
    t_in_window = np.nanmean(np.diff(xx[_window_]))

    return t_forward, t_back, t_forward-_cent, _cent-t_back, N_thresh_1, N_in_dip, t_in_window


def calculate_integral(x0, y0, yerr0, R, S):
    """Calculate the integral and integral error of a light curve.
    
    Parameters:
    -----------
    x0 (array-like): Time values of the light curve dip.
    y0 (array-like): Magnitude values of the light curve dip.
    yerr0 (array-like): Magnitude errors of the light curve dip.
    R (float): Biweight location of the light curve (global).
    S (float): Biweight scale of the light curve (global).

    Returns:
    --------
    integral (float): Integral of the light curve.
    integral_error (float): Integral error of the light curve.
    """
    # Integral equation
    #TODO: check that this calculation is right.
    integral = np.sum((y0[1::] - R) * (np.diff(x0)/2))
    integral_error = np.sum((yerr0[1::]**2 + S**2) * (np.diff(x0)/2)**2)
    
    return integral, np.sqrt(integral_error)

def calculate_assymetry_score(Int_left, Int_right, Int_err_left, Int_err_right):
    """Calculate the assymetry score of a light curve using the GP mean curve minmum as the zero-phase of the identified dip.
    
    Parameters:
    -----------
    Int_left (float): Integral of the light curve to the left of the center.
    Int_right (float): Integral of the light curve to the right of the center.
    Int_err_left (float): Integral error of the light curve to the left of the center.
    Int_err_right (float): Integral error of the light curve to the right of the center.

    Returns:
    --------
    assymetry_score (float): Assymetry score of the light curve.
    """
    # Assymetry score
    assymetry_score = (Int_left - Int_right) / (np.sqrt(Int_err_left**2 + Int_err_right**2))
    
    return assymetry_score  


def peak_detector(times, dips, power_thresh=3, peak_close_rmv=15, pk_2_pk_cut=30):
    """
    Run and compute dip detection algorithm on a light curve.
    
    Parameters:
    -----------
    times (array-like): Time values of the light curve.
    dips (array-like): Deviation values of the light curve.
    power_thresh (float): Threshold for the peak detection. Default is 3.
    peak_close_rmv (float): Tolerance for removing peaks that are too close to each other. Default is 15.
    pk_2_pk_cut (float): Minimum peak to peak separation. Default is 30 days.

    Returns:
    --------
    N_peaks (int): Number of peaks detected.
    dip_summary (dict): Summary of the dip. Including the peak location, the window start and end, the number of 1 sigma detections in the dip, the number of detections in the dip, the forward and backward duration of the dip, and the dip power.
    """
    try:
        if len(dips)==0:
            return 0, 0

        #TODO: add smoothing savgol_filter again...
        yht = dips

        # Scipy peak finding algorithm
        pks, _ = find_peaks(yht, height=power_thresh, distance=pk_2_pk_cut) #TODO: is 100 days peak separation too aggresive?

        # Reverse sort the peak values
        pks = np.sort(pks)[::-1]
        
        # Time of peaks and dev of peaks
        t_pks, p_pks = times[pks], dips[pks]
        
        # Number of peaks
        N_peaks = len(t_pks)
        
        dip_summary = {}
        for i, (time_ppk, ppk) in enumerate(zip(t_pks, p_pks)):
            #TODO: old version
            #_edges = calc_dip_edges(times, dips, time_ppk, atol=0.2)
            _edges = detect_bursts_edges(times, dips, time_ppk, np.nanmean(dips), np.nanstd(dips), burst_threshold=3.0, expansion_indices=1)
            # t_start, t_end, abs(t_start-center_time), abs(t_end-center_time), N_thresh_1, 0, 0 : above. #TODO: remove this!
            
            dip_summary[f'dip_{i}'] = {
                "peak_loc": time_ppk,
                'window_start': _edges[0],
                'window_end': _edges[1],
                "N_1sig_in_dip": _edges[-3], # number of 1 sigma detections in the dip
                "N_in_dip": _edges[-3], # number of detections in the dip
                'loc_forward_dur': _edges[2],
                "loc_backward_dur": _edges[3],
                "dip_power":ppk,
                "average_dt_dif": _edges[-1]
            }
                    
        return N_peaks, dip_summary
    except:
        return 0, 0


def best_peak_detector(peak_dictionary, min_in_dip=1):
    """Chose the best peak from the peak detector with a minimum number of detections threshold. 
    
    Parameters:
    -----------
    peak_dictionary (dict): Dictionary of the peaks.
    min_in_dip (int): Minimum number of detections in the dip. Default is 3 detections.

    Returns:
    --------
    pd.DataFrame: Table of the best dip properties.
    """
    # unpack dictionary
    N_peaks, dict_summary = peak_dictionary
    
    summary_matrix = np.zeros(shape=(N_peaks, 9)) # TODO: add more columns to this matrix
    for i, info in enumerate(dict_summary.keys()):
       summary_matrix[i,:] = np.array(list(dict_summary[f'{info}'].values()))

    dip_table = pd.DataFrame(summary_matrix, columns=['peak_loc', 'window_start', 'window_end', 'N_1sig_in_dip', 'N_in_dip', 'loc_forward_dur', 'loc_backward_dur', 'dip_power', 'average_dt_dif'])

    #TODO: we can remove this eventually...
    # From previous analysis we have found that the following locations are bad... please ignore such alias effect.
    bad_loc_lower = [158248.52098,
                    158261.52098,
                    158448.52098,
                    158576.52098,
                    158834.52098,
                    158854.52098,
                    158855.52098,
                    158879.02098,
                    159266.02098,
                    159301.52098,
                    159448.52098]

    bad_loc_upper = [158249.52098,
                    158262.52098,
                    158449.52098,
                    158577.52098,
                    158835.52098,
                    158855.52098,
                    158856.52098,
                    158880.02098,
                    159267.02098,
                    159302.52098,
                    159449.52098]

    # Search in the table if none of these pair ranges exist; if so then remove 
    bad_loc_indices = []

    # Check if any pair of loc_lower and loc_upper intersects with the specified bounds
    for i, (lower, upper) in enumerate(zip(bad_loc_lower, bad_loc_upper)):
        condition = (dip_table['peak_loc'].between(lower, upper)) | (dip_table['peak_loc'].between(upper, lower))
        if any(condition):
            bad_loc_indices.extend(dip_table.index[condition].tolist())

    dip_table = dip_table.drop(bad_loc_indices) # drop aliasing times

    dip_table_q = dip_table['N_in_dip'] >= min_in_dip # must contain at least one detection at the bottom

    try:
        if len(dip_table_q) == 0:
            #print ("No dip is found within the minimum number of detections.")
            return None
        
        elif len(dip_table_q)==1:
            return dip_table 
        else: # TODO: `dip_power` or `N_in_dip`
            return pd.DataFrame([list(dip_table.iloc[dip_table[dip_table_q]['N_in_dip'].idxmax()])], columns=['peak_loc', 'window_start', 'window_end', 'N_1sig_in_dip', 'N_in_dip', 'loc_forward_dur', 'loc_backward_dur', 'dip_power', 'average_dt_dif'])
    except:
        return None