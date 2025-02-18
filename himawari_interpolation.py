"""
Interpolate data from the Himawari dataset using Inverse-Distance-Weighted Interpolation with KDTree.

Originally written by denis-bz.

Adapted by tomarmstro
"""

from __future__ import division
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import get_thredds_file
from datetime import datetime
import matplotlib.pyplot as plt
from config import CONFIG
import os
import invdisttree


def interogate_himawari_csv(site_name):
    """
    Loads or initializes Himawari satellite data for a specified site and returns relevant data details.

    This function checks for an existing CSV file containing Himawari satellite data for the given site.
    If the file is found, it loads the data and sets the `last_himawari_data_date` to the most recent date in the file.
    If the file does not exist, it initializes an empty dataframe with relevant columns and sets `last_himawari_data_date`
    to the first available Himawari data date, as specified in the configuration.

    Args:
        site_name (str): The name of the site for which Himawari data is being queried. Used to locate the file.

    Returns:
        tuple:
            - last_himawari_data_date (datetime): The date of the latest Himawari data entry if available; otherwise, 
                                                  the first date of available Himawari data (e.g., 2019-04-01).
            - file_interpolation_output (DataFrame): A dataframe containing either the loaded Himawari data or an initialized 
                                                     empty structure with required columns.
            - site_himawari_data_filename (str): The file path for the Himawari data file for the specified site.
    """
    # Specify filename for himawari data of this specific site
    proc_file_path = CONFIG["PROCESSED_FILE_PATH"]
    site_himawari_data_filename = rf"{proc_file_path}\himawari_interpolation\{site_name}_himawari_results.csv"

    # Check for saved himawari data and load it if exists.
    if os.path.exists(site_himawari_data_filename):
        site_himawari_data = pd.read_csv(site_himawari_data_filename, index_col=0)
        last_himawari_data_date = pd.to_datetime(site_himawari_data.iloc[-1]['date'])
        print("A Himawari data file for this site exists! Loading site_himawari_data_filename...")
        file_interpolation_output = site_himawari_data.copy()
        file_interpolation_output['date'] = pd.to_datetime(file_interpolation_output['date']).dt.tz_convert('Etc/GMT-10')
    # Otherwise, initialize the empty dataframe
    else: 
        print("No himawari data file found, one will be created.")
        file_interpolation_output = pd.DataFrame(columns = [
        'date', 'target_latitude', 'target_longitude', 'interpolated_value (Mj m-2 hr-1)',
        'interpolated_value (umol m-2 s-1)', 'basic_interpolated_value', 'interpolated_value_difference', 'file_url',
        'N', 'Ndim', 'Nask', 'Nnear', 'leafsize', 'eps', 'p', 'cycle', 'seed'])
        # If no himawari csv, set the 'last_himawari_data_date' to the first himawari data date on thredds server (2019/04/01)
        last_himawari_data_date = datetime.strptime(CONFIG['FIRST_HIMAWARI_DATA_DATE'], "%Y-%m-%d")

    return last_himawari_data_date, file_interpolation_output, site_himawari_data_filename


# def main(site_name, latitude, longitude, start_time, end_time):
def main(site_name, latitude, longitude):
    """
    Main function to process and interpolate Himawari satellite data for a specified site location.

    This function retrieves, processes, and interpolates Himawari satellite irradiance data files for the given
    site's latitude and longitude coordinates. It handles data filtering, interpolation, and updates an output
    CSV file with processed values. If the data file already contains a record, it skips the file; otherwise, 
    it performs interpolation and saves new results. Progress is saved every 100 processed files.

    Args:
        site_name (str): The name of the site for which data is being processed, used to locate and save data files.
        latitude (float): Latitude of the site for spatial filtering and interpolation.
        longitude (float): Longitude of the site for spatial filtering and interpolation.

    Returns:
        DataFrame: A dataframe containing interpolated data for the specified site, including irradiance values,
                   file URLs, and interpolation details.
    """
    script_start_time = datetime.now()
    
    last_himawari_data_date, file_interpolation_output, site_himawari_data_filename = interogate_himawari_csv(site_name)

    # get thredds url
    file_urls, file_count = get_thredds_file.main(last_himawari_data_date)

    file_counter = 0
    processed_file_counter = 0
    
    # Iterate through all of our catalog's files
    for file_url in file_urls:
        file_url = file_url.replace('fileServer', 'dodsC')
        # if the url is already cached, skip it
        if file_interpolation_output['file_url'].str.contains(file_url, na=False).any():
            print(f"File URL match found, skipping file: {file_url}.")
            file_counter += 1
            continue

        # otherwise, get the interpolated values    
        else:
            print(f"\nProcessing {file_url}..")

            lat_max = latitude + CONFIG['FILTER_DEGREES']
            lat_min = latitude - CONFIG['FILTER_DEGREES']
            lon_max = longitude + CONFIG['FILTER_DEGREES']
            lon_min = longitude - CONFIG['FILTER_DEGREES']

            ds = xr.open_dataset(file_url, decode_times = False)
            file_date = datetime.strptime(file_url[-15:-3], '%Y%m%d%H%M')

            # Filter the data by coordinate to reduce interpolation load - Not sure if this helps?
            filtered_ds = ds.where(
                (ds['latitude'] > lat_min) &
                (ds['latitude'] < lat_max) &
                (ds['longitude'] > lon_min) &
                (ds['longitude'] < lon_max),
                drop=True)
            
            data = filtered_ds[CONFIG['DATA_INTERVAL'] + '_integral_of_surface_global_irradiance'][0].values

            # Convert coordinates into useable format
            x = filtered_ds['longitude'].values
            y = filtered_ds['latitude'].values

            interpolated_value, interpolated_value_converted_umol, basic_interpolated_value, \
                interpolated_value_difference, interp_features = invdisttree.run_invdisttree_interpolation(data, x, y, longitude, latitude)

            # Add the new row of data to the dataframe
            row_to_add = pd.DataFrame([[file_date, latitude, longitude, interpolated_value, interpolated_value_converted_umol,
                basic_interpolated_value, interpolated_value_difference, file_url,
                interp_features['N'], interp_features['Ndim'], interp_features['Nask'], interp_features['Nnear'], interp_features['leafsize'], 
                interp_features['eps'], interp_features['p'], interp_features['cycle'], interp_features['seed']]], columns=file_interpolation_output.columns)
            # Convert the date for the row to be added to GMT-10
            row_to_add['date'] = row_to_add['date'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-10')
            file_interpolation_output = pd.concat([file_interpolation_output, row_to_add], ignore_index=True)
            file_counter += 1
            print(f"{file_counter}/{file_count} files processed.")
            
            processed_file_counter += 1
            if processed_file_counter == 100:
                update_himawari_csv(file_interpolation_output, site_himawari_data_filename)
                processed_file_counter = 0
                last_himawari_data_date, file_interpolation_output, site_himawari_data_filename = interogate_himawari_csv(site_name)
            
    # Update himawari csv with final remaining data
    update_himawari_csv(file_interpolation_output, site_himawari_data_filename)
    # Interogate the final updates himawari csv to pass to main par corrections script
    last_himawari_data_date, file_interpolation_output, site_himawari_data_filename = interogate_himawari_csv(site_name)
    print('Interpolation took: {}'.format(datetime.now() - script_start_time))
    return file_interpolation_output

            
def update_himawari_csv(file_interpolation_output, site_himawari_data_filename):
    # Convert the date to datetime and the appropriate timezone for main par corrections script
    file_interpolation_output['date'] = pd.to_datetime(file_interpolation_output['date']).dt.tz_convert('Etc/GMT-10')
    # Sort by date to ensure things are in order
    file_interpolation_output = file_interpolation_output.sort_values(by='date').reset_index(drop=True)
    # saving the dataframe
    file_interpolation_output.to_csv(site_himawari_data_filename)
    print(f"Saved himawari interpolation data as {site_himawari_data_filename}.")
    return file_interpolation_output


if __name__ == "__main__":
    site_name = "Davies_test"
    latitude = -19.13866
    longitude = 146.899
    main(site_name, latitude, longitude)