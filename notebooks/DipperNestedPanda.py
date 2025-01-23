import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask
import astropy.units as u
import lsdb
import dask.array
import dask.distributed
import light_curve as licu
import nested_pandas as npd
from dask_expr import from_legacy_dataframe
from lsdb import read_hipscat
from nested_dask import NestedFrame
from dask.distributed import Client
from evaluate import evaluate_updated

def main():
    # load ZTF object table
    ztf = lsdb.read_hipscat("/epyc/data3/hipscat/catalogs/ztf_axs/ztf_dr14", 
                            columns=['ps1_objid', 'ra', 'dec'])

    # Load ZTF DR17 ZUBERBCAL sources
    ztf_sources = lsdb.read_hipscat("/epyc/data3/hipscat/catalogs/zubercal", 
                                columns=["mjd", "mag", "magerr", "band", "flag", "fieldid", "objectid", "rcidin", 
                                        "objra", "objdec", "info"])

    # Define light curve columns
    lc_columns = ["mjd", "mag", "magerr", "band", "flag", "info", "fieldid", "objectid", "rcidin", "objra", "objdec"]

    print ("Initializing DASK client")
    client = Client(n_workers=10, memory_limit="95GiB", threads_per_worker=1)
    print (client)

    print ("..Loading object STARHORSE FGK object...")
    hips_fgk = lsdb.read_hipscat("/nvme/users/atzanida/tmp/StarHorseApril21_output_hips")

    # Select only relevant columns
    final_selected_fgk = hips_fgk[['RA_ICRS_StarHorse', 'DE_ICRS_StarHorse', 'ps1_objid_ztf_dr14', 
                                    'teff50_StarHorse', 'GMAG0_StarHorse']]
    
    # Box selection on the sky!
    slab_1 = final_selected_fgk.box_search(ra=(30, 250), dec=(10, 45))
    hips_fgk = slab_1 # rename...

    def convert_to_nested_frame(df: pd.DataFrame, nested_columns: list[str]):
        """Map a pd.DataFrame to a Nested Pandas."""
        
        other_columns = [col for col in df.columns if col not in nested_columns]

        # Since object rows are repeated, we just drop duplicates
        object_df = df[other_columns].groupby(level=0).first()
        nested_frame = npd.NestedFrame(object_df)

        source_df = df[nested_columns]
        
        # lc is for light curve
        return nested_frame.add_nested(source_df, "lc")
    
    # join hips FGK object table to ZTF sources
    lsdb_joined = hips_fgk.join(
        ztf_sources,
        left_on="ps1_objid_ztf_dr14",
        right_on="objectid",
        suffixes=("", ""),
    )

    # get joined dataframe
    joined_ddf = lsdb_joined._ddf

    # map partitions of jointed dataframe
    ddf = joined_ddf.map_partitions(
        lambda df: convert_to_nested_frame(df, nested_columns=lc_columns),
        meta=convert_to_nested_frame(joined_ddf._meta, nested_columns=lc_columns),
    )

    nested_ddf = NestedFrame.from_dask_dataframe(ddf)

    # Select only catflags=0 and r-band detections
    nested_ddf_update = nested_ddf.query("lc.flag == 0 and lc.info == 0 ")

    # feature evaluation 
    column_names = ['Nphot',
        'biweight_scale',
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
    
    feature_table = nested_ddf_update.reduce(evaluate_updated,
                                         "lc.mjd", 
                                         "lc.mag", 
                                         "lc.magerr",
                                         "lc.flag",
                                         "lc.band",
                                         meta={name: np.float64 for name in column_names})

    print (".............Now storing the data as a parquet file!!")
    feature_table.to_parquet("/nvme/users/atzanida/tmp/output_zubercal_dipper_search_smallv1.parquet")


if __name__ == "__main__":
    # make sure we time the operation
    import time
    start = time.time()
    main()
    end = time.time()
    print ("Total time taken: ", end-start)




