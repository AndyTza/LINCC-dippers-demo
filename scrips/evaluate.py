from dipper import *
from tools import *
import astropy.stats as astro_stats
from scipy.stats import median_abs_deviation
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')

# TODO: undo this action
import warnings
warnings.filterwarnings('ignore')

# feature evaluation 
column_names = ['Nphot',
    'biweight_scale',
    'frac_above_2_sigma', # in deviation
    'Ndips',
    'rate',
    'chi2dof',
    'skew', 
    'kurtosis',
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

def half_eval(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, ra_cat, dec_cat, custom_cols=column_names, min_phot=10):
    """Perform half evaluation of the light curve."""
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat,  band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Evaluate biweight location and scale & other obvious statistics
    R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
    adf = adf_tests(mag) # ADF test for stationarity
    chi2dof = chidof(mag) # chi2dof

    # Running deviation
    running_deviation = deviation(mag, mag_err, R, S)

    # Peak detection summary per light curve
    peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=1, pk_2_pk_cut=1)

    return peak_detections


def evaluate_only_dips(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, custom_cols=column_names):
    """ Evaluate the time-series features of the light curve, as of May 28, 2024. 

    Parameters
    ----------
    time_cat : array-like
        Time array of the light curve.
    mag_cat : array-like
        Magnitude array of the light curve.
    mag_err_cat : array-like
        Magnitude error array of the light curve.
    flag_cat : array-like
        Flag array of the light curve.
    band_cat : array-like
        Band array of the light curve (supports ZTF-r or ZTF-g).
    custom_cols : list
        List of custom column names.
    
    Returns
    -------
    pd.Series
        A pandas series containing the evaluated features.

    General Notes
    -------------
    This evaluation function takes in the ZTF (gr) detections and performs a dipper selection function.  
        
    """

    # Summary information
    summary_ = {}
    passing = False
    
    # Digest my light curve. Select band, good detections & sort
    time0, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, 
                                    band_of_study='r', flag_good=0, q=None, custom_q=False)
    mag_err  = mag_err/10000 # flux error to mag error (relevant for ZUBERCAL only.)
    
    # Don't evaluate if there are less than 10 detections
    if (len(time0) < 10):
        passing = True 
    else:
        time, mag = bin_counter(time0, mag, 2) # running median filter
        bla, mag_err = bin_counter(time0, mag_err, 2) # running median filter
        
        # clean up the photometry 
        _filter = (~np.isnan(mag)) & (~np.isnan(mag_err))
        time, mag, mag_err = time[_filter], mag[_filter], mag_err[_filter]

        # Evaluate biweight location and scale & other obvious statistics
        R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
        chi2dof = chidof(mag) # chi2dof

        # Running deviation
        running_deviation = deviation(mag, mag_err, R, S)

        # Peak detection summary per light curve
        peak_detections = peak_detector(time, running_deviation, power_thresh=4, peak_close_rmv=50, pk_2_pk_cut=50)

        # Calculate other summary statistics
        other_stats = other_summary_stats(time, mag, mag_err, len(mag), R, S)
            
        # If there's no detected peaks or time array is empty or no peaks detected...
        if peak_detections[0]==0 or len(time)==0 or peak_detections[0]==0:
            # skiping the nan case.
            passing = True

        else: # If there are significant peaks...
            # From the r-band data select a good peak...
            bp = best_peak_detector(peak_detections, min_in_dip=3)
            
            # Investigate the g-band data and ensure we see a ~significant~ event 
            g_validate, out_g = False, 0
            
            time_g0, mag_g, mag_err_g = prepare_lc(time_cat, mag_cat, mag_err_cat,
                                                   flag_cat, band_cat, band_of_study='g', flag_good=0, q=None, custom_q=False)
            time_g, mag_g = bin_counter(time_g0, mag_g, 3) # running median filter
            bla, mag_err_g = bin_counter(time_g0, mag_err_g, 3) # running median filter

            mag_err_g = mag_err_g/10000

            clean_g = ~np.isnan(mag_g) & ~np.isnan(mag_err_g)

            time_g, mag_g, mag_err_g = time_g[clean_g], mag_g[clean_g], mag_err_g[clean_g]
            
            # minimum number of g-band detections after processing
            if len(time_g) > 10:
                g_validate = True
                
            Rg, Sg = astro_stats.biweight.biweight_location(mag_g), astro_stats.biweight.biweight_scale(mag_g)
            
            running_deviation_g = deviation(mag_g, mag_err_g, Rg, Sg)

            # TODO: major big in the dipper detection pipeline is not as efficient here!
            try:
                best_peak_time = bp['peak_loc'].values[0]
                close_g_dets = np.isclose(best_peak_time, time_g, atol=30) # Selection find nearest point within 20 days

                if sum(close_g_dets)==0:
                    g_validate = False
                    xg = [] # empty array...
                else:
                    close_pair_time = time_g[close_g_dets]
                    close_pair_mag = mag_g[close_g_dets]
                    close_pair_mag_err = mag_err_g[close_g_dets]
                    close_pair_dev = running_deviation_g[close_g_dets]

                    phase_close_g = abs(best_peak_time-close_pair_time) # closest neighbour
                    srt_pairs = np.argsort(phase_close_g) # first index will be the nearest pair to best peak time!!

                    _time_close_g, _mag_close_g, _magerr_close_g, _running_deviation_g = close_pair_time[srt_pairs], close_pair_mag[srt_pairs], close_pair_mag_err[srt_pairs], close_pair_dev[srt_pairs]

                    # final selection!!
                    close_pair_time = _time_close_g[0]
                    close_pair_mag = _mag_close_g[0]
                    close_pair_mag_err = _magerr_close_g[0]
                    close_pair_dev = _running_deviation_g[0]
                    xg = [0]
            except:
                g_validate = False
                xg = [] # empty array...
            
            if (len(xg) == 0) or (g_validate==False): # reject if there's no detections...
                g_validate = False
            else:
                g_validate = True
                # Calculate the significance of this g-band bump...
                out_g = (close_pair_dev-np.nanmean(running_deviation_g))/(np.nanstd(running_deviation_g))
        
            # 3 sigma deviation cut...
            if g_validate and out_g >2.5: # both r-band and g-band data show similar peaks...
        
                _score_ = calc_sum_score(time, mag, mag_err, peak_detections, R, S)

                # If failing; set all values to NaN
                for col in custom_cols:
                    summary_[col] = np.nan

                ######## Final appending data ########
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                summary_['chi2dof'] = chi2dof
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']
                summary_['best_dip_power'] = bp['dip_power'].values[0]
                summary_['best_dip_time_loc'] = bp['peak_loc'].values[0]
                summary_['best_dip_start'] = bp['window_start'].values[0]
                summary_['best_dip_end'] = bp['window_end'].values[0]
                summary_['best_dip_dt'] = bp['average_dt_dif'].values[0]
                summary_['best_dip_ndet'] = bp['N_in_dip'].values[0]
                summary_['lc_score'] = _score_
            
            else:
               passing = True

    if passing==False:
        return dict(summary_)
    elif passing==True:
        pass


def evaluate_updated(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, custom_cols=column_names):
    """ Evaluate the time-series features of the light curve, as of May 28, 2024. Current pipeline as of Aug. 2024.

    Parameters
    ----------
    time_cat : array-like
        Time array of the light curve.
    mag_cat : array-like
        Magnitude array of the light curve.
    mag_err_cat : array-like
        Magnitude error array of the light curve.
    flag_cat : array-like
        Flag array of the light curve.
    band_cat : array-like
        Band array of the light curve (supports ZTF-r or ZTF-g).
    custom_cols : list
        List of custom column names.
    
    Returns
    -------
    pd.Series
        A pandas series containing the evaluated features.

    General Notes
    -------------
    This evaluation function takes in the ZTF (gr) detections and performs a dipper selection function.  
        
    """
    # Summary information
    summary_ = {}
    
    # make into .values
    #time_cat, mag_cat, mag_err_cat, flag_cat, band_cat = time_cat.values, mag_cat.values, mag_err_cat.values, flag_cat.values, band_cat.values
    
    # Digest my light curve. Select band, good detections & sort
    time0, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, 
                                    band_of_study='r', flag_good=0, q=None, custom_q=False)
    #print (len(time0))
    #print (time0)

    mag_err  = mag_err/10000 # flux error to mag error

    # Don't evaluate if there are less than 10 detections
    if (len(time0) < 10):
        summary_['Nphot'] = len(time0)
        for col in custom_cols[1::]:
            summary_[col] = np.nan
    else:
        time = time0

        # TODO: pause here and check if the bin_counter is working as expected
        #time, mag = bin_counter(time0, mag, 2) # running median filter
        #bla, mag_err = bin_counter(time0, mag_err, 2) # running median filter
        
        # clean up the photometry 
        _filter = (~np.isnan(mag)) & (~np.isnan(mag_err))
        time, mag, mag_err = time[_filter], mag[_filter], mag_err[_filter]

        # Evaluate biweight location and scale & other obvious statistics
        # TODO: swaping with medians and std
        R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)

        chi2dof = chidof(mag) # chi2dof

        # Running deviation
        running_deviation = deviation(mag, mag_err, R, S)

        # Peak detection summary per light curve TODO: updated sigma to 3
        peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=50, pk_2_pk_cut=50)

        # Calculate other summary statistics
        other_stats = other_summary_stats(time, mag, mag_err, len(mag), R, S)
            
        # If there's no detected peaks or time array is empty or no peaks detected...
        if peak_detections[0]==0 or len(time)==0 or peak_detections[0]==0:
            # If failing; set all values to NaN
            for col in custom_cols:
                summary_[col] = np.nan
            
            # Replace nan's with values
            summary_['Nphot'] = len(time)
            summary_['biweight_scale'] = S

            if len(running_deviation)==0:
                summary_['frac_above_2_sigma'] = 0
            else:
                summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
            
            summary_['Ndips'] = 0
            summary_['rate'] = 0
            summary_['chi2dof'] = chi2dof
            summary_['skew'] = other_stats['skew']
            summary_['kurtosis'] = other_stats['kurtosis']
            summary_['stetson_i'] = other_stats['stetson_I']
            summary_['stetson_j'] = other_stats['stetson_J']
            summary_['stetson_k'] = other_stats['stetson_K']
            summary_['invNeumann'] = other_stats['invNeumann']
        else: # If there are significant peaks...

            # From the r-band data select a good peak...
            bp = best_peak_detector(peak_detections, min_in_dip=3)
            
            # Investigate the g-band data and ensure we see a ~significant~ event 
            g_validate, out_g = False, 0
            
            time_g0, mag_g, mag_err_g = prepare_lc(time_cat, mag_cat, mag_err_cat,
                                                   flag_cat, band_cat, band_of_study='g', flag_good=0, q=None, custom_q=False)
            time_g = time_g0
            #TODO: check speed here.
            #time_g, mag_g = bin_counter(time_g0, mag_g, 2) # running median filter
            #bla, mag_err_g = bin_counter(time_g0, mag_err_g, 2) # running median filter

            mag_err_g = mag_err_g/10000

            clean_g = ~np.isnan(mag_g) & ~np.isnan(mag_err_g)

            time_g, mag_g, mag_err_g = time_g[clean_g], mag_g[clean_g], mag_err_g[clean_g]

            # minimum number of g-band detections after processing
            if len(time_g) > 10:
                g_validate = True

            # TODO: median and std are not as sensitive!
            #Rg, Sg = np.median(mag_g), np.std(mag_g)    
            Rg, Sg = astro_stats.biweight.biweight_location(mag_g), astro_stats.biweight.biweight_scale(mag_g)
            
            running_deviation_g = deviation(mag_g, mag_err_g, Rg, Sg)

            # TODO: major big in the dipper detection pipeline is not as efficient here!
            try:
                best_peak_time = bp['peak_loc'].values[0]
                #close_g_dets = np.where((time_g>best_peak_time-20) & (time_g<best_peak_time+20))[0] # TODO: new!
                #print (close_g_dets)
                close_g_dets = np.isclose(best_peak_time, time_g, atol=30) # Selection find nearest point within 30 days

                if sum(close_g_dets)==0:
                    g_validate = False
                    xg = [] # empty array...
                else:
                    close_pair_time = time_g[close_g_dets]
                    close_pair_mag = mag_g[close_g_dets]
                    close_pair_mag_err = mag_err_g[close_g_dets]
                    close_pair_dev = running_deviation_g[close_g_dets]

                    phase_close_g = abs(best_peak_time-close_pair_time) # closest neighbour
                    srt_pairs = np.argsort(phase_close_g) # first index will be the nearest pair to best peak time!!

                    _time_close_g, _mag_close_g, _magerr_close_g, _running_deviation_g = close_pair_time[srt_pairs], close_pair_mag[srt_pairs], close_pair_mag_err[srt_pairs], close_pair_dev[srt_pairs]
                    #plt.figure(figsize=(10, 5))
                    #plt.scatter(_time_close_g, _mag_close_g-np.median(mag_g), s=50, color='purple', marker='s')
                 
                    # if there's many why don't we take the average?
                    if np.any(close_pair_dev>2): # when to take the median vs. when to take the closest value?
                        at_max = np.argmax(_running_deviation_g)
                        close_pair_time = _time_close_g[at_max]
                        close_pair_mag = _mag_close_g[at_max]
                        close_pair_mag_err = _magerr_close_g[at_max]
                        close_pair_dev = _running_deviation_g[at_max]
                    else:
                        # else take the first index
                        close_pair_time = _time_close_g[0]
                        close_pair_mag = _mag_close_g[0]
                        close_pair_mag_err = _magerr_close_g[0]
                        close_pair_dev = _running_deviation_g[0]

                    xg = [0]
                    #plt.figure(figsize=(25, 5))
                    #plt.scatter(time, mag-np.median(mag), s=10, color='red')
                    #plt.scatter(time_g, mag_g-np.median(mag_g), s=20, color='green')
                    #plt.scatter(close_pair_time, close_pair_mag-np.median(mag_g), s=30, color='k', marker='x')
                    #print (abs(close_pair_time-best_peak_time))
                    #plt.axvline(best_peak_time)
                    #for a in time_g[close_g_dets]:
                    #    plt.axvline(a, alpha=0.5, lw=0.5)
                    #plt.xlim(best_peak_time-850, best_peak_time+850)
            except:
                g_validate = False
                xg = [] # empty array...
            
            if (len(xg) == 0) or (g_validate==False): # reject if there's no detections...
                g_validate = False
            else:
                g_validate = True
                # Calculate the significance of this g-band bump...
                # TODO: changed the nanmean to nanmedian... seems to make large ampltiude dips more easily to detect.
                #out_g = (close_pair_dev-np.nanmean(running_deviation_g))/(np.nanstd(running_deviation_g))
                out_g = (close_pair_dev-astro_stats.biweight_location(running_deviation_g))/(astro_stats.biweight_scale(running_deviation_g))
                #print (out_g)
            # 3 sigma deviation cut... # TODO: updated the out_g condition to 1 sigma.
            #print ("outg", out_g)
            if g_validate and out_g > 1.5: # both r-band and g-band data show similar peaks...
        
                _score_, minX, maxX, MM = calc_sum_score(time, mag, mag_err, peak_detections, R, S, out_g)
                
                # try re-computing the significance of out_G with respect to the FWHM bounds...
                close_g_dets = np.where((time_g>minX) & (time_g<maxX))[0] # TODO: new!
                #print (running_deviation_g[close_g_dets])
                
                ########################### REPEAT ALGORITHM for Gband detections!!... ######
                if sum(close_g_dets)==0 and peak_detections[0]>1:
                    g_validate = False
                    xg = [] # empty array...
                    close_pair_dev_atmax = close_pair_dev
                else:
                    close_pair_time = time_g[close_g_dets]
                    close_pair_mag = mag_g[close_g_dets]
                    close_pair_mag_err = mag_err_g[close_g_dets]
                    close_pair_dev = running_deviation_g[close_g_dets]
                 
                    # if there's many why don't we take the average?
                    if np.any(close_pair_dev>2): # when to take the median vs. when to take the closest value?
                        at_max = np.argmax(_running_deviation_g)
                        close_pair_time = _time_close_g[at_max]
                        close_pair_mag = _mag_close_g[at_max]
                        close_pair_mag_err = _magerr_close_g[at_max]
                        #print ("ff2")
                        # take the max if there are at least >=2 dets above 2
                        # or else take the median
                        
                        N2s = len(close_pair_dev[close_pair_dev>=2])
                        #print (">>>", close_pair_dev, N2s)
                        if N2s>=2:
                            #print ("max")
                            close_pair_dev_atmax = np.max(close_pair_dev) # max 
                        else: #select smallest...
                            if np.log10(_score_)>3:
                               #print ("Alt")
                                close_pair_dev_atmax = np.max(close_pair_dev)
                            else:  
                                #print ("med")
                                close_pair_dev_atmax = np.min(close_pair_dev)
                    else:
                        #print ("default close")
                        # else take the first index
                        close_pair_time = _time_close_g[0]
                        close_pair_mag = _mag_close_g[0]
                        close_pair_mag_err = _magerr_close_g[0]
                        if np.log10(_score_)>4:
                            close_pair_dev_atmax = np.median(_running_deviation_g) # MEDIAN 
                        else:
                            close_pair_dev_atmax = np.mean(_running_deviation_g)# MEAN
                        #print ("DF M")
                        
                out_g_FINALE = (close_pair_dev_atmax-astro_stats.biweight_location(running_deviation_g))/(astro_stats.biweight_scale(running_deviation_g))
                #print ("OUT G INITIAL:", out_g)
                #print ("OUT G FINALE:", out_g_FINALE) 
                
                #print ('..........FINAL PASSER: ', out_g_FINALE, out_g_FINALE>2)
                                                                                             
                                                                                         
                # If failing; set all values to NaN
                for col in custom_cols:
                    summary_[col] = np.nan

                ######## Final appending data ########
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = MM # abs model - mean data (r-band) DELTA MAG
                summary_['chi2dof'] = chi2dof
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']
                summary_['best_dip_power'] = bp['dip_power'].values[0]
                summary_['best_dip_time_loc'] = bp['peak_loc'].values[0]
                summary_['best_dip_start'] = bp['window_start'].values[0]
                summary_['best_dip_end'] = bp['window_end'].values[0]
                summary_['best_dip_dt'] = int(out_g_FINALE>2) # THIS IS THE G-R strength above 2 SIGMA!
                summary_['best_dip_ndet'] = bp['N_in_dip'].values[0]
                summary_['lc_score'] = _score_
            
            else:
                # If failing; set all values to NaN
                for col in custom_cols:
                    summary_[col] = np.nan
                    
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                summary_['chi2dof'] = chi2dof
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']
    try:
        return dict(summary_)
    except:
        # return a summary with nans
        return dict({'Nphot': np.nan,
                      'biweight_scale': np.nan, 
                      'frac_above_2_sigma': np.nan, 
                      'Ndips': np.nan, 'rate': np.nan, 
                      'chi2dof': np.nan, 'skew': np.nan,
                        'kurtosis': np.nan, 'stetson_i': np.nan, 'stetson_j': np.nan, 'stetson_k': np.nan, 'invNeumann': np.nan, 'best_dip_power': np.nan, 'best_dip_time_loc': np.nan, 'best_dip_start': np.nan, 'best_dip_end': np.nan, 'best_dip_dt': np.nan, 'best_dip_ndet': np.nan, 'lc_score': np.nan})



