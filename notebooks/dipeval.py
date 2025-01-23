from dipper import *
from tools import *
import astropy.stats as astro_stats
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')

# TODO: undo this action
import warnings
warnings.filterwarnings('ignore')

# feature evaluation 
column_names = ['biweight_scale',
    'frac_above_2_sigma', # in deviation
    'Ndips',
    'rate',
    'chi2dof',
    'skew', 
    'kurtosis',
    'mad',
    'stetson_i',
    'stetson_j',
    'stetson_k',
    'invNeumann',    
    'best_dip_power',
    'best_dip_time_loc',
    'best_dip_start',
    'best_dip_end',
    'best_dip_dt',
    'best_dip_ndet',
    'lc_score']


def light_curve_quality(time_cat, mag_cat, mag_err_cat, flag_cat):
    """Return the number of detections in the light curve when applying some quality cuts.
    
    Parameters
    ----------
    time_cat : array
        Time of the observations.
    mag_cat : array
        Magnitude of the observations.
    mag_err_cat : array
        Error in the magnitude of the observations.
    flag_cat : array
        Flag of the observations."""
    try:
        time, mag, mag_err = time_cat, mag_cat, mag_err_cat

        # Remove bad things.
        rmv = (mag_err>0) & (~np.isnan(time)) & (~np.isnan(mag)) & (~np.isnan(mag_err)) 

        time, mag, mag_err = time[rmv], mag[rmv], mag_err[rmv]

        # Remove consecutive observations
        ts = abs(time - np.roll(time, 1)) > 1e-5
        time, mag, mag_err = time[ts], mag[ts], mag_err[ts]
    
        return {"Nphot": len(time)}
    except:
        return {"Nphot":0}
    
def has_initial_dips(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat):
    
    try:
        # Digest my light curve. Select band, good detections & sort
        time0, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, 
                                        band_of_study='r', flag_good=0, q=None, custom_q=False)
        mag_err  = mag_err/10000 # flux error to mag error (relevant for ZUBERCAL only.)

        time, mag = bin_counter(time0, mag, 2) # running median filter
        bla, mag_err = bin_counter(time0, mag_err, 2) # running median filter
        
        # clean up the photometry 
        _filter = (~np.isnan(mag)) & (~np.isnan(mag_err))
        time, mag, mag_err = time[_filter], mag[_filter], mag_err[_filter]

        # Evaluate biweight location and scale & other obvious statistics
        R, S = np.median(mag), np.std(mag)

        # Running deviation
        running_deviation = deviation(mag, mag_err, R, S)

        # Peak detection summary per light curve
        peak_detections = peak_detector(time, running_deviation, power_thresh=4, peak_close_rmv=50, pk_2_pk_cut=50)

        return {"Ndips" : peak_detections[0]}
    except:
        return {"Ndips" : 0}
