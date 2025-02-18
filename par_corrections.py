"""
Main file for AIMS historical weather station surface PAR dataset corrections.

Author: Thomas Armstrong (tomarmstro)
Created: 1/03/2024
"""

# imports
from sklearn import datasets
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
from scipy.stats import zscore
import math
from datetime import date
from statistics import StatisticsError
import par_algos
import par_plots
import himawari_interpolation
import os
from datetime import datetime
import argparse

# config
from config import CONFIG

"""
TODO: Implement daytime filter - Either a time threshold, zenith angle or incorporate Vinny's code?
TODO: Adjust the file intake - using config for the input file path is clunky - consider a basic file selection gui?
TODO: Better visualisations - Assess if it is worth using plotly for better interactivity of plots (par_plots) - Which plots to produce routinely
TODO: Validation of outputs?
TODO: Should solar noon be calculated with raw or model par?
TODO: Can we screen shadows out of daily data - Look at the ratio of raw par to model par to identify shadows according to some cutoff?
TODO: Check the new csv for Cloudless days with outlier days removed - Make the output section neater regarding this
TODO: Is the pratio correction based on the entire deployment? Does missing data in a deployment cause issues here?
"""

def main(input_file):
    """
    Main function to process, analyze, and save PAR data with correction and visualization.

    This script coordinates the execution of all processing steps, including setting up data, 
    fetching and interpolating Himawari satellite data, applying PAR corrections, and saving the 
    output. It also generates plots to visualize the processed data for quality assurance.

    Workflow:
        1. Initializes data setup and retrieves site-specific details.
        2. Prepares the daytime dataset and fetches Himawari data for interpolation.
        3. Corrects PAR data using daytime dataset and deployment start dates.
        4. Saves the processed datasets and generates plots for visualization.

    """

    # Check if the provided file exists
    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        return

    script_start_time = datetime.now()
    df, site_name = data_setup(input_file)
    # df, site_name = data_setup()
    print(f"Processing data from {site_name}..")
    # site_name_slice = site_name[0:6]
    daytime_df, deployment_start_dates = daytime_dataset_setup(df)
    daytime_df = get_himawari_data(df, site_name, daytime_df)
    final_daytime_model_df, final_daily_df, final_cloudless_df, final_filtered_cloudless_df = get_corrected_par(deployment_start_dates, daytime_df, site_name)
    save_outputs(site_name, final_daytime_model_df, final_daily_df, final_cloudless_df, final_filtered_cloudless_df)
    # Build plots
    par_plots.build_plots(site_name, final_daytime_model_df, final_daily_df, final_cloudless_df, deployment_start_dates)
    print(f'Processing {site_name} took: {(datetime.now() - script_start_time)}')
    

def get_himawari_data(df, site_name, daytime_df):
    """
    Retrieves and interpolates Himawari satellite data from the NCI THREDDS server and merges it with provided daytime data.

    This function accesses hourly Himawari satellite data available from 01/04/2019 onwards, filling data gaps through 
    linear interpolation. Interpolation is performed using Inverse-Distance-Weighted (IDW) interpolation with KDTree 
    as implemented in `indisttree.py`. The resulting data, which includes converted photosynthetically active radiation 
    (PAR) values, is then merged into a filtered DataFrame containing daytime values.

    Args:
        df (DataFrame): Input DataFrame containing latitude, longitude, times, and initial PAR data.
        site_name (str): Name of the site for which data is being retrieved.
        daytime_df (DataFrame): DataFrame of filtered daytime values with PAR data to be merged with Himawari data.

    Returns:
        DataFrame: The updated `daytime_df` containing merged Himawari interpolated data with 10-minute resampling.
    
    Raises:
        TypeError: If Himawari data is unavailable or if merging fails, an error message is printed.

    """
    # Get himawari data
    himawari_interpolated = himawari_interpolation.main(site_name, df['latitude'][0], df['longitude'][0])
    try:
        # Select only the converted par data and time
        himawari_interpolated = himawari_interpolated[['date', 'interpolated_value (Mj m-2 hr-1)', 'interpolated_value (umol m-2 s-1)', 'basic_interpolated_value'
]]

        # Add the himawari data to the daily dataframe
        daytime_df = pd.merge(daytime_df, himawari_interpolated, on='date', how='left')

        # Resample the himawari data to get data for every 10 minutes
        # final_daytime_model_df['himawari_resampled'] = resample_10_min(final_daytime_model_df[['date', 'interpolated_value (umol m-2 s-1)']])
        daytime_df = pd.merge(daytime_df, resample_10_min(daytime_df[['date', 'interpolated_value (umol m-2 s-1)']]), on='date', how='left')

    except TypeError as error:
        # No himawari data available
        print(f"No himawari data found: {error}")
        pass
    return daytime_df


def resample_10_min(data):
    """
    Resamples Himawari data to 10-minute intervals and linearly interpolates to fill gaps.

    Args:
        data (DataFrame): Input DataFrame with Himawari PAR data at hourly intervals.

    Returns:
        df_interpolated (DataFrame): Resampled and interpolated DataFrame containing 10-minute interval data with the 
                   'himawari_resampled' column.
    """

    # Sample DataFrame (with time in hourly increments)
    df = pd.DataFrame(data)

    # Convert 'time' column to datetime format and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Resample to 10-minute intervals
    df_resampled = df.resample('10T').asfreq()

    # Interpolate the missing values (linear)
    df_interpolated = df_resampled.interpolate(method='linear')
    df_interpolated = df_interpolated.rename(columns={'interpolated_value (umol m-2 s-1)': 'himawari_resampled'})

    # Reset the index
    df_interpolated = df_interpolated.reset_index()

    return df_interpolated[['date', 'himawari_resampled']]


def data_setup(input_file):
    """
    Performs initial data wrangling on raw PAR data and associated metadata.

    This function reads PAR data from a specified input file, processes columns, and prepares 
    metadata by adjusting data types, handling timezones, and formatting timestamps. It extracts 
    key date and time components for easier data manipulation and analysis, renames certain columns, 
    and sorts the data chronologically by date.

    Returns:
        tuple: A tuple containing:
            - data (DataFrame): A DataFrame of processed PAR data.
            - site_name (str): The name of the site, formatted to replace spaces with underscores.
    """

    print("Setting up data..")
    # df = pd.read_csv(CONFIG["INPUT_FILE"])
    df = pd.read_csv(input_file)
    site_name = df['site_name'][0].replace(" ", "_")
    # Convert date to local time - Is this necessary?
    df['date'] = pd.to_datetime(df['time']).dt.tz_convert('Etc/GMT-10')
    # Adjusted by 10 minutes?
    # df['date'] = df['date'] - timedelta(minutes=10)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['instrument_serial_no'] = df['serial_num']
    df = df.rename(columns={"raw_value": "rawpar"})
    df.sort_values(by='date', inplace=True)
    data = df.copy()
    return data, site_name


def daytime_dataset_setup(data):
    """
    Prepares a dataset containing only daytime PAR (Photosynthetically Active Radiation) values and 
    associated metadata, including zenith angle calculations and deployment dates.

    This function filters the provided data to isolate daytime observations, calculates zenith angles 
    based on latitude, longitude, and time, and creates additional columns with astronomical metadata 
    (e.g., zenith angle, radius vector, equation of time, and declination) to support light-related 
    analyses. It also identifies deployment start and end dates based on changes in instrument serial 
    numbers, recording each deployment period.

    Args:
        data (DataFrame): A DataFrame containing initial wrangled PAR data with date and time components.

    Returns:
        tuple: 
            - day_time_df (DataFrame): A DataFrame of daytime PAR data and metadata, including 
              zenith angle and deployment timing.
            - deployment_start_dates (list): A list of dates marking the start of each deployment period.
    """

    print("Extracting daytime data..")

    # Filter by time (All data between two specific times/hours)
    # day_time_df = data.loc[(data['hour'] >= 5.0) | (data['hour'] <= 19.0)].copy()

    day_time_df = data.copy()
    
    # Replaces daynumber function (Get the start year, month, day)
    dnstart = date(int(day_time_df['year'].iloc[0]), int(day_time_df['month'].iloc[0]),
                   int(day_time_df['day'].iloc[0])).timetuple().tm_yday

    # Initiate lists for zenith and model_par
    z_func_list = []
    dn1_list = []
    dn_list = []

    # Get zenith from zen()
    for i in range(len(day_time_df)):
        # zenith angle z estimated. if z lt 0 skip
        # Days into the year
        dn = date(int(day_time_df['year'].iloc[i]), int(day_time_df['month'].iloc[i]),
                  int(day_time_df['day'].iloc[i])).timetuple().tm_yday
        dn_list.append(dn)
        z = par_algos.zen(day_time_df['latitude'].iloc[0] * math.pi / 180.0, day_time_df['longitude'].iloc[0],
                dn, day_time_df['hour'].iloc[i], day_time_df['minute'].iloc[i])
        z_func_list.append(z)
        del1 = day_time_df['year'].iloc[i] - day_time_df['year'].iloc[0]
        # Dayseq algorithm
        # Days into deployment
        dn1 = 365 - dnstart + ((del1 - 1) * 365) + dn
        dn1_list.append(dn1)

    # Build zenith lists and add to df
    z_list = []
    rv_list = []
    et_list = []
    dec_list = []
    for z in z_func_list:
        z_list.append(z[0])
        rv_list.append(z[1])
        et_list.append(z[2])
        dec_list.append(z[3])
    day_time_df['zenith_angle'] = z_list
    day_time_df['radius_vec'] = rv_list
    day_time_df['equ_time'] = et_list
    day_time_df['declination'] = dec_list
    day_time_df['dn'] = dn_list
    day_time_df['dn1'] = dn1_list

    # Set df column names
    day_time_df = day_time_df[['date', 'dn1', 'dn', 'day', 'month', 'year',
                               'hour', 'minute', 'instrument_serial_no', 'zenith_angle',
                               'radius_vec', 'equ_time', 'declination', 'rawpar']]

    dn1_counter = 0
    deployment_start_dates = []

    # Get first date of dataset (Start of first deployment)
    current_instrument = day_time_df['instrument_serial_no'].iloc[0]
    deployment_start_dates.append(day_time_df['date'].dt.date.iloc[0])
    # Check for new instrument id which indicates a new deployment
    print("Detecting change of instruments..")
    for current_date, day in day_time_df.groupby(day_time_df['date'].dt.date):
        if day['instrument_serial_no'].iloc[0] != current_instrument and not pd.isna(day['instrument_serial_no'].iloc[0]):
            print(f"Instrument changed from {current_instrument} to {day['instrument_serial_no'].iloc[0]} on {current_date}!")
            current_instrument = day['instrument_serial_no'].iloc[0]
            dn1_counter = 0
            deployment_start_dates.append(current_date)

        else:
            dn1_counter += 1

    # Get last date of dataset (End of last deployment)
    deployment_start_dates.append(day_time_df['date'].dt.date.iloc[-1])

    # Display all deployment start/end dates
    print("Deployment start dates: ", deployment_start_dates)
    return day_time_df, deployment_start_dates


def get_deployments(day_time_df):
    """
    Extracts the start dates of all instrument deployments based on daily sequence markers.

    This function identifies the start of each deployment period in the provided dataset, where 
    deployments are marked by a specific sequence value ('dn1' = 0) that indicates the first day 
    of a new deployment. Deployment start dates are recorded and returned as a list, which can be 
    used to track deployment periods for data analysis or reporting.

    Args:
        day_time_df (DataFrame): A DataFrame containing daytime PAR data, with deployment day numbers (dn1) 
                                 and date information.

    Returns:
        deployment_start_dates (list): A list of dates marking the start of each deployment period.
    """

    deployment_start_dates = []
    for i in range(len(day_time_df)):
        if day_time_df['dn1'].iloc[i] == 0:
            deployment_start_dates.append(day_time_df[date].iloc[0])
            print(f"Start of a deployment noted at {day_time_df[date].iloc[0]}.")

    return deployment_start_dates


def get_modpar_oldcorpar(daytime_model_df):
    """
    Computes modelled and corrected Photosynthetically Active Radiation (PAR) values for daytime observations.

    This function builds a model of expected PAR values based on zenith angle and corrects the raw PAR data 
    using an empirical ratio ("old" method). It applies a quality control (QC) flag to identify observations 
    where the model-to-observed PAR ratio exceeds a configured threshold. 

    Args:
        daytime_model_df (DataFrame): DataFrame containing daytime PAR observations with associated metadata 
                                      such as zenith angle and radius vector.

    Returns:
        daytime_model_df_updated (Dataframe): Updated DataFrame with appended columns for modelled PAR (`modpar`), corrected PAR 
                   (`old_corpar`), deployment day sequence (`dn1`), and QC flag (`qc_flag`). 
    """

    print("Calculating model PAR..")

    old_corpar_list = []
    modpar_list = []
    dn1_update_list = []
    dn1_update_counter = 1
    qc_flag_list = []
    # Get model_par
    for i in range(len(daytime_model_df)):
        modpar = par_algos.get_model_par(daytime_model_df['zenith_angle'].iloc[i])
        # Correct for sun-earth distance (radius vector)
        modpar = modpar * daytime_model_df['radius_vec'].iloc[i]  # Placeholder for rv value
        modpar_list.append(modpar)

        # Should this 'dn1/dn' value be the days into deployment??
        pratio = 0.0001221 * daytime_model_df['dn1'].iloc[i] + 0.95767

        # Old method of getting corrected PAR.
        old_corpar = daytime_model_df['rawpar'].iloc[i] * pratio
        old_corpar_list.append(old_corpar)
        dn1_update_list.append(daytime_model_df['dn1'].iloc[i]-daytime_model_df['dn1'].iloc[0]+1)
        dn1_update_counter += 1
        # daytime_model_df['dn1'] = daytime_model_df['dn1']-daytime_model_df['dn1'].iloc[0]

        raw_mod_par_ratio = modpar / daytime_model_df['rawpar'].iloc[i]

        if abs(raw_mod_par_ratio) >= CONFIG['RATIO_THRESHOLD']:
            qc_flag = 0
        else: 
            qc_flag = 1
        qc_flag_list.append(qc_flag)
        
    daytime_model_df_updated = daytime_model_df.copy()
    daytime_model_df_updated['dn1'] = dn1_update_list
    daytime_model_df_updated['modpar'] = modpar_list
    daytime_model_df_updated['old_corpar'] = old_corpar_list
    daytime_model_df_updated['qc_flag'] = qc_flag_list

    return daytime_model_df_updated


def get_filtered_tilt(df):
    """
    Calculates tilt values by comparing the timing of peak observed PAR to peak modelled PAR within a midday window.

    This function assesses potential tilt in the sensor setup by determining the time shift between 
    observed (raw) and modelled peak PAR values during midday (typically between 11 AM and 1 PM). 
    A time shift between the peaks suggests possible tilt, which this function records as the difference 
    in 10-minute intervals between maximum observed and modelled PAR values. 

    Args:
        df (DataFrame): DataFrame containing raw and modelled PAR data with associated metadata.

    Returns:
        delineation_vals_list (list): List of time differences in 10-minute intervals between maximum observed and modelled PAR values 
              for each day in the dataset. If no midday data is available, a tilt value of 0 is assigned.
    """
    delineation_vals_list = []
    for current_date, day in df.groupby(df['date'].dt.date):

        # Restricting tilt calculation to the middle of the day (typically 11-13 hrs)
        tilt_df = day.loc[(day['hour'] >= CONFIG["TILT_START_TIME"]) & (day['hour'] < CONFIG["TILT_END_TIME"])].copy()
        midday_rawpar = tilt_df['rawpar']
        midday_modpar = tilt_df['modpar']
        try:
            index_of_daily_max_rawpar = np.argmax(midday_rawpar)
            index_of_daily_max_modpar = np.argmax(midday_modpar)
            # Tilt value = how many 10 minute periods does the max rawpar differ from max modpar
            delineation_val = index_of_daily_max_modpar - index_of_daily_max_rawpar
            delineation_vals_list.append(delineation_val)
        # If no noon values exist (ie instrument recovered early that day), set tilt to 0
        except ValueError as error:
            print("Error raised:", error, "- This means no midday values exist to calculate tilt on: ",
                  current_date, "- Tilt is set to 0")
            delineation_val = 0
            delineation_vals_list.append(delineation_val)
            continue

    return delineation_vals_list


def get_raw_tilt(df):
    """
    Calculates tilt values based on the timing difference between peak observed PAR and peak modelled PAR values.

    This function identifies potential sensor tilt by determining the offset in 10-minute intervals 
    between the maximum raw PAR value (observed) and the maximum modelled PAR value over a full day. 
    Unlike filtered tilt calculations, this method assesses tilt without limiting to midday hours.

    Args:
        df (DataFrame): DataFrame containing raw and modelled PAR data with associated metadata.

    Returns:
        raw_delineation_vals_list (list): List of time differences in 10-minute intervals between maximum observed and modelled PAR values 
              for each day in the dataset. Positive values suggest modelled peaks occur after observed peaks, 
              indicating possible tilt.
    """

    raw_delineation_vals_list = []
    for day in df.groupby(df['date'].dt.date):
        # Tilt values without middle of the day restriction
        raw_delineation_val = np.argmax(day['modpar']) - np.argmax(day['rawpar'])
        raw_delineation_vals_list.append(raw_delineation_val)
    return raw_delineation_vals_list


def build_daily_cloudless_df(daytime_model_df, raw_tilt, filtered_tilt, abs_tilt):
    """
    Condenses daytime PAR data to daily values and filters for cloudless days based on tilt and clear-sky criteria.

    This function compiles daily summaries of PAR data by aggregating key statistics (e.g., noon PAR ratios, 
    total daily PAR) and calculating tilt indicators for each day. A cloudless flag is then applied to days 
    that meet specified clear-sky conditions, enabling isolation of cloud-free data for further analysis.

    Args:
        daytime_model_df (DataFrame): DataFrame containing raw and modelled PAR data with additional 
                                      metadata, filtered to exclude nighttime data.
        raw_tilt (list): List of raw tilt values calculated without midday filtering.
        filtered_tilt (list): List of tilt values with midday filtering.
        abs_tilt (ndarray): Array of absolute tilt values between raw and model PAR peaks.

    Returns:
        tuple: A tuple containing:
            - daily_df (DataFrame): DataFrame of daily PAR data summaries with associated tilt and cloudless flags.
            - cloudless_df (DataFrame): DataFrame of daily data filtered to include only cloudless days, 
                                        with PAR statistics and tilt information.
    """

    print("Building daily and cloudless dataframes..")
    daily_df_list = []
    daily_df_cloudless = []
    cloudless_list = []
    ratiop_list = []
    cloudless_flag_list = [0] * len(daytime_model_df)
    daytime_model_df['cloudless_flag'] = cloudless_flag_list
    cloudless_dates = []
    counter = 0

    for day in daytime_model_df.groupby(daytime_model_df['date'].dt.date):
        try:
            # Filter daytime values by zenith value
            # filtered_day = day[(day['zenith_angle'] >= 0) & (day['zenith_angle'] <= 90)]
            const, const1 = par_algos.statsxy(day['rawpar'], day['modpar'])
        except StatisticsError:
            # only one par value left in the day = can't get variance, so skip this day
            ratiop_list.append(np.nan)
            continue

        # Do we call noon at the highest par for raw or model??
        noon_rawpar_index = np.argmax(day['rawpar'])
        noon_modpar_index = np.argmax(day['modpar'])
        noon_rawpar = day['rawpar'].iloc[noon_rawpar_index]
        noon_modpar = day['modpar'].iloc[noon_modpar_index]

        noon_himawari_index = np.argmax(day['himawari_resampled'])
        noon_himawari = day['himawari_resampled'].iloc[noon_himawari_index]

        noon_par_ratio = noon_rawpar / noon_modpar

        sum_rawpar = (600.0 / 10 ** 6) * np.sum(day['rawpar'])
        sum_modpar = (600.0 / 10 ** 6) * np.sum(day['modpar'])

        # Collate all clear stats data
        daily_df_list.append((day['date'].dt.date.iloc[0], day['dn1'].iloc[0], day['dn'].iloc[0],
                              day['day'].iloc[0], day['month'].iloc[0], day['year'].iloc[0]) +
                             tuple(const1[:19]) + (sum_rawpar, sum_modpar, noon_rawpar, noon_modpar,
                                                   raw_tilt[counter], filtered_tilt[counter],
                                                   abs_tilt[counter], noon_himawari))

        # Select for cloudless days
        # if any values in const1 between [5] and [16] are less than the CLOUDLESS_THRESHOLD, we have a cloudless day, so append it to the list
        if any(const1[i] <= CONFIG["CLOUDLESS_THRESHOLD"] for i in range(5, 16)):
            ratio_sum_par = sum_modpar / sum_rawpar
            ratio_noon_par = noon_modpar / noon_rawpar
            cloudless_list.append((day['date'].dt.date.iloc[0], day['dn1'].iloc[0], day['dn'].iloc[0],
                                   day['day'].iloc[0], day['month'].iloc[0], day['year'].iloc[0],
                                   sum_rawpar, sum_modpar, ratio_sum_par,
                                   noon_rawpar, noon_modpar, ratio_noon_par,
                                   raw_tilt[counter], filtered_tilt[counter],
                                   abs_tilt[counter], noon_himawari))
            daily_df_cloudless.append(1)
            cloudless_dates.append(day['date'].iloc[0].date())
        else:
            daily_df_cloudless.append(0)
        cloudless_days = []
        if day['date'].iloc[0].date() in cloudless_dates:
            cloudless_days.append(1)
        else:
            cloudless_days.append(0)
        counter += 1
    daily_df = pd.DataFrame(daily_df_list, columns = ['date', 'dn1', 'dn', 'day', 'month', 'year', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6',
                        '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8',
                        '1.9', 'sum_rawpar', 'sum_modpar', 'noon_rawpar', 'noon_modpar',
                        'raw_tilt', 'filtered_tilt', 'abs_tilt', 'noon_himawari'])

    # Build cloudless dataframe and set column names
    cloudless_df = pd.DataFrame(cloudless_list, columns = ['date', 'dn1', 'dn', 'day', 'month', 'year', 'sum_rawpar', 'sum_modpar', 'ratio_sum_par',
                            'noon_rawpar', 'noon_modpar', 'ratio_noon_par', 'raw_tilt', 'filtered_tilt', 'abs_tilt', 'noon_himawari'])
    # Add cloudless flags to clear_stats
    daily_df['cloudless'] = daily_df_cloudless

    return daily_df, cloudless_df


def get_consecutive_tilt(daily_df, cloudless_df):
    """
    Identifies periods of consecutive tilt in the PAR data to detect potential instrument misalignment or movement.

    This function analyzes tilt values and identifies sequences where the tilt remains above a certain threshold 
    for an extended period, which could indicate that the instrument may have fallen or shifted. It then calculates 
    a rolling average of tilt for smoother data interpretation and adds consecutive tilt flags to the daily DataFrames.

    Args:
        daily_df (DataFrame): DataFrame containing daily PAR values and associated tilt data for all days.
        cloudless_df (DataFrame): DataFrame containing daily PAR values and tilt data for cloudless days only.

    Returns:
        tuple: A tuple containing:
            - daily_tilt_df (DataFrame): DataFrame with a column indicating whether consecutive tilt was detected 
                                         and a rolling average of tilt over 5 days.
            - cloudless_tilt_df (DataFrame): DataFrame with a rolling average of tilt for cloudless days only.
    """

    print("Getting consecutive tilt values..")
    # Find consecutive tilt values and flag
    consecutive_tilt_count = 0
    max_consecutive_tilt_count = 4  # Number of consecutive tilt values we check for
    max_consecutive_tilt_list = []
    for value in daily_df['abs_tilt']:
        try:
            # if daily_df['abs_tilt'].loc[i] >= 5:  # Tilt value we check for
            if value >= 5:
                consecutive_tilt_count += 1
                if consecutive_tilt_count >= max_consecutive_tilt_count:
                    max_consecutive_tilt_list.append(1)
                else:
                    max_consecutive_tilt_list.append(0)
            else:
                consecutive_tilt_count = 0
                max_consecutive_tilt_list.append(0)
        except ValueError as error:
            print(error)
            consecutive_tilt_count = 0
            max_consecutive_tilt_list.append(0)
    daily_tilt_df = pd.DataFrame()
    cloudless_tilt_df = pd.DataFrame()
    daily_tilt_df['tilted'] = max_consecutive_tilt_list
    daily_tilt_df['tilt_rolling_avg'] = daily_df['filtered_tilt'].rolling(5).mean()
    cloudless_tilt_df['tilt_rolling_avg'] = cloudless_df['filtered_tilt'].rolling(5).mean()
    return daily_tilt_df, cloudless_tilt_df


def get_consecutive_cloudless(daily_df):
    """
    Identifies and flags sequences of consecutive cloudless days in the data, indicating whether 
    the number of consecutive cloudless days exceeds a specified threshold.

    This function scans through the `cloudless` column of the provided DataFrame and counts 
    consecutive cloudless days. If the count exceeds a predefined threshold, it flags those days 
    with a `1`, otherwise, it flags them with a `0`.

    Args:
        daily_df (DataFrame): DataFrame containing daily PAR values, including a `cloudless` column
                               where `1` indicates a cloudless day and `0` indicates a non-cloudy day.

    Returns:
        DataFrame: A DataFrame with a new column, `consec_clouds`, that contains a `1` for each 
                   day that is part of a sequence of consecutive cloudless days exceeding the threshold, 
                   and `0` otherwise.
    """

    # Find consecutive cloudless days and flag
    consecutive_cloudless_count = 0
    max_consecutive_cloudless_count = 4  # Number of consecutive cloudless days we check for
    max_consecutive_cloudless_list = []
    for value in daily_df['cloudless']:
        if value >= 1:  # Cloudless or not - 1 for cloudless
            consecutive_cloudless_count += 1
            if consecutive_cloudless_count >= max_consecutive_cloudless_count:
                max_consecutive_cloudless_list.append(1)
            else:
                max_consecutive_cloudless_list.append(0)
        else:
            consecutive_cloudless_count = 0
            max_consecutive_cloudless_list.append(0)
    consec_cloudless_df = pd.DataFrame()
    consec_cloudless_df['consec_clouds'] = max_consecutive_cloudless_list
    return consec_cloudless_df


def get_correction_coeffs(cloudless_df):
    """
    Calculates corrected PAR (photosynthetically active radiation) coefficients by fitting a polynomial model 
    to cloudless day data and filtering out outliers based on residuals' standard deviation. It compares the 
    polynomial fit to a linear model to determine the best fit for correction, then applies the selected coefficients 
    to compute corrected PAR values.

    The function first performs a polynomial fit on the provided data and calculates residuals. Outliers, based on 
    the standard deviation of residuals, are excluded, and the polynomial fit is recalculated using the filtered data. 
    A linear regression model is also applied for comparison. The best polynomial fit is chosen based on which has 
    coefficients closest to those expected (i.e., coefficient values closer to 1). The corrected PAR values are then 
    calculated using the selected polynomial coefficients.

    Args:
        cloudless_df (DataFrame): DataFrame of daily PAR values excluding cloudy days, with columns 
                                  for 'ratio_sum_par', 'dn1', and 'noon_rawpar'.

    Returns:
        tuple: 
            - cloudless_df (DataFrame): Input DataFrame with additional columns for corrected PAR values, 
                                        polynomial coefficients, and linear model coefficients.
            - used_coeffs (List): List containing the coefficients used for the correction (from the best-fit polynomial).
            - filtered_cloudless_df (DataFrame): DataFrame containing only cloudless data points that passed the 
                                                  outlier filtering based on standard deviation.
    """

    # Correcting PAR
    # Create empy lists
    corrected_ratio_list = []
    corrected_par_list = []
    # Additional cloudless filter (Anything over a certain raw-model par ratio can't be cloudless
    cloudless_df = cloudless_df.drop(cloudless_df[cloudless_df['ratio_noon_par'] >= CONFIG['RATIO_THRESHOLD']].index)
    # Variables for test
    # y = cloudless_df['ratio_noon_par']
    y = cloudless_df['ratio_sum_par']
    x = cloudless_df['dn1']
    std_deviation_tolerance = 2
    degrees = 1
    # Fit initial polynomial test
    coeffs = np.polyfit(x, y, degrees, full=True)
    fitted_curve = np.polyval(coeffs[0], x)
    # Get residuals (fitted data - raw data)
    residuals = y - fitted_curve
    # Get z_scores (Number of standard deviations away)
    z_scores = zscore(residuals)
    # Get indicies of values within standard deviation tolerance
    filtered_indices = np.abs(z_scores) <= std_deviation_tolerance
    # Get data within standard deviation tolerance/at filterered indicies (remove outliers)
    y_filt = y[filtered_indices]
    x_filt = x[filtered_indices]
    # Rerun polynomial test with filtered data
    coeffs_filt = np.polyfit(x_filt, y_filt, degrees, full=True)
    # Correction for zenith angle assumed to be 1
    zenith_correction = 1

    print(f"First pass correction coefficients are {coeffs[0][0]} and {coeffs[0][1]}.")
    print(f"Second pass correction coefficients are {coeffs_filt[0][0]} and {coeffs_filt[0][1]}.")
    if abs(1 - coeffs[0][1]) <= abs(1 - coeffs_filt[0][1]):
        used_coeffs = coeffs[0]
        used_coeffs_pass = "First"
    else:
        used_coeffs = coeffs_filt[0]
        used_coeffs_pass = "Second"

    # Get corrected ratio and corrected par with filtered coefficients
    for i in range(len(cloudless_df['dn1'])):
        # For all filtered values, calculate corrected ratio
        if np.abs(z_scores).iloc[i] <= std_deviation_tolerance:
            corrected_ratio = used_coeffs[0] * cloudless_df['dn1'].iloc[i] + used_coeffs[1] * zenith_correction
        else:
            # If noon raw par is very low/high
            corrected_ratio = np.nan
        corrected_ratio_list.append(corrected_ratio)
        # Use corrected ratio to calculate corrected par
        corrected_par = cloudless_df['noon_rawpar'].iloc[i] * corrected_ratio
        corrected_par_list.append(corrected_par)
    print(f"{used_coeffs_pass} pass PAR correction coefficients being used: {used_coeffs[0]} and {used_coeffs[1]}.")
    # # Linear Model?
    x = np.array([x]).reshape(-1,1)
    linear_model = LinearRegression().fit(x, y)
    linear_model.fit(x, y)
    r_sq = linear_model.score(x, y)
    print(f"Linear model correction coefficients are {linear_model.coef_[0]} and {linear_model.intercept_}.")
    print(f"Difference between polynomial and linear model first pass coefficients: {linear_model.coef_[0] - coeffs[0][0]} and {linear_model.intercept_ - coeffs[0][1]}")
    # print(f"Linear model coefficient of determination: {r_sq}.")

    cloudless_df['corrected_ratio'] = corrected_ratio_list
    cloudless_df['corrected_par'] = corrected_par_list
    
    cloudless_df['coeff1'] = used_coeffs[0]
    cloudless_df['coeff2'] = used_coeffs[1]

    # Add linear coefficients for comparison
    cloudless_df['linear_coeff1'] = linear_model.coef_[0]
    cloudless_df['linear_coeff2'] = linear_model.intercept_

    # Add difference between polynomial and linear for comparison
    cloudless_df['poly_linear__dif_coeff1'] = linear_model.coef_[0] - coeffs[0][0]
    cloudless_df['poly_linear__dif_coeff2'] = linear_model.intercept_ - coeffs[0][1]

    # Filtered cloudless days
    filtered_cloudless_df = pd.DataFrame(cloudless_df.loc[filtered_indices])

    return cloudless_df, used_coeffs, filtered_cloudless_df


def get_corrected_par(deployment_start_dates, decamin_df, site_name):
    """
    Processes PAR (photosynthetically active radiation) data for multiple deployments, 
    including the calculation of corrected PAR values, tilt corrections, cloudless day identification, 
    and coefficient adjustments. This function aggregates and processes data for daytime, daily, 
    and cloudless day PAR values, applying correction algorithms and saving the final outputs.

    The function iterates through multiple deployments, filtering the data by zenith angle and 
    correcting for tilt and cloudless days. For each deployment, it calculates corrected PAR coefficients 
    using a polynomial fit, applies the corrections to the data, and appends the results to the final dataframes.
    Additionally, it handles edge cases like insufficient cloudless days by skipping deployments that do not meet 
    the required minimum threshold.

    Args:
        deployment_start_dates (list): A list of deployment start dates (as strings), where each entry 
                                       represents the beginning of a new deployment period.
        decamin_df (DataFrame): A dataframe containing the PAR data, including date, zenith angle, and raw PAR 
                                values, for both day and night periods.
        site_name (str): The name of the site for which the data is being processed. This is used for naming output files.

    Returns:
        tuple:
            - final_daytime_model_df (DataFrame): A dataframe containing corrected PAR values for 10-minute daytime samples.
            - final_daily_df (DataFrame): A dataframe with daily PAR values, corrected for tilt and cloudless days.
            - final_cloudless_df (DataFrame): A dataframe with corrected PAR values for cloudless days only.
            - final_filtered_cloudless_df (DataFrame): A dataframe with cloudless days data filtered for outliers based on residuals.
    """
    blank = []
    # Create empty dataframes
    final_daytime_model_df = pd.DataFrame(blank)
    final_daily_df = pd.DataFrame(blank)
    final_cloudless_df = pd.DataFrame(blank)
    final_filtered_cloudless_df = pd.DataFrame(blank)

    # Filter by zenith
    daytime_decamin_df = decamin_df[(decamin_df["zenith_angle"] > 0) & (decamin_df["zenith_angle"] < 90)].copy()
    
    for i in range(len(deployment_start_dates)-1):
        print("\n**************************************************\
               \n***************** New Deployment *****************\
               \n**************************************************")
        deployment_start_date = str(deployment_start_dates[i])
        print("Deployment Start: ", deployment_start_date)
        deployment_end_date = str(deployment_start_dates[i+1])
        print("Deployment End: ", deployment_end_date)
        # Split data into deployments
        single_deployment_daytime_mask = (daytime_decamin_df['date'] > deployment_start_date) & (daytime_decamin_df['date'] <= deployment_end_date)
        single_deployment_daytime_df = daytime_decamin_df.loc[single_deployment_daytime_mask]
        # Get model par and old corpar and add to single_deployment_daytime_data df
        single_deployment_daytime_df = get_modpar_oldcorpar(single_deployment_daytime_df)

        # Get tilt values
        raw_tilt = get_raw_tilt(single_deployment_daytime_df)
        filtered_tilt = get_filtered_tilt(single_deployment_daytime_df)
        abs_tilt = np.abs(filtered_tilt)

        # Get daily and cloudless dfs
        daily_df, cloudless_df = build_daily_cloudless_df(single_deployment_daytime_df, raw_tilt, filtered_tilt, abs_tilt)

        # Get tilt and cloudless flags - add to dfs
        daily_tilt_df, cloudless_tilt_df = get_consecutive_tilt(daily_df, cloudless_df)
        daily_df = daily_df.join(daily_tilt_df)
        cloudless_df = cloudless_df.join(cloudless_tilt_df)

        # Check for having enough Cloudless days
        # Create variable for config to enable use in f-string
        config_min_cloudless = CONFIG["MINIMUM_CLOUDLESS_DAYS"]
        if len(cloudless_df) < config_min_cloudless:
            print(f"Less than {config_min_cloudless} cloudless days in this deployment, moving to next deployment.")
            continue
        daily_df = daily_df.join(get_consecutive_cloudless(daily_df))

        # Get corrected par coefficients
        print("Calculating corrected PAR..")
        cloudless_df, coeffs, filtered_cloudless_df = get_correction_coeffs(cloudless_df)

        # Add the unfiltered by zenith rows (ie. night time data) - But only within this deployment
        nighttime_df = decamin_df[
        (~decamin_df['date'].isin(single_deployment_daytime_df['date'])) & 
        (decamin_df['date'] >= deployment_start_date) & 
        (decamin_df['date'] <= deployment_end_date) ]

        # concatenate the new rows to df1
        single_deployment_daytime_df = pd.concat([single_deployment_daytime_df, nighttime_df], ignore_index=True)

        # Sort by date to get readded nighttime values in order
        single_deployment_daytime_df = single_deployment_daytime_df.sort_values(by='date').reset_index(drop=True)
        
        # Main corrected PAR algorithm
        single_deployment_daytime_df['corpar'] = (coeffs[0] * single_deployment_daytime_df['dn1'] + coeffs[1]) * single_deployment_daytime_df['rawpar']
        single_deployment_daytime_df['coeff1'] = coeffs[0]
        single_deployment_daytime_df['coeff2'] = coeffs[1]

        daily_df['noon_corpar'] = (coeffs[0] * daily_df['dn1'] + coeffs[1]) * daily_df['noon_rawpar']
        daily_df['coeff1'] = coeffs[0]
        daily_df['coeff2'] = coeffs[1]

        deployment_duration_str = f"{deployment_start_date}_{deployment_end_date}"
        save_outputs(site_name, single_deployment_daytime_df, daily_df, cloudless_df, filtered_cloudless_df, deployment_duration_str)

        final_daytime_model_df = pd.concat([final_daytime_model_df, single_deployment_daytime_df], ignore_index=True)

        final_daily_df = pd.concat([final_daily_df, daily_df], ignore_index=True)

        final_cloudless_df = pd.concat([final_cloudless_df, cloudless_df], ignore_index=True)
        final_filtered_cloudless_df = pd.concat([final_filtered_cloudless_df, filtered_cloudless_df], ignore_index=True)
        
    return final_daytime_model_df, final_daily_df, final_cloudless_df, final_filtered_cloudless_df


def save_outputs(site_name, daytime_model_df, daily_df, cloudless_df, filtered_cloudless_df, deployment_duration_str = "all_deployments"):
    """
    Saves the processed PAR output dataframes as CSV files and creates the corresponding output plots. 

    This function takes the final processed dataframes for different PAR values (daytime, daily, cloudless, and filtered cloudless), 
    and saves them as CSV files in a specified directory. It also ensures that the output directory exists before saving the files.

    Args:
        site_name (str): The name of the site, used for naming the output files and organizing them into site-specific directories.
        daytime_model_df (DataFrame): A dataframe containing corrected PAR values for 10-minute daytime samples (pcorrA).
        daily_df (DataFrame): A dataframe containing daily PAR values (clear_stats).
        cloudless_df (DataFrame): A dataframe containing corrected PAR values for cloudless days only (daily samples).
        filtered_cloudless_df (DataFrame): A dataframe containing filtered cloudless day PAR values based on residuals.
        deployment_duration_str (str, optional): A string representing the duration of the deployment, used to customize the file name 
                                                 (default is "all_deployments"). It allows users to differentiate between different 
                                                 deployment periods when saving files.

    Returns:
        None: The function does not return any values. It saves the dataframes to CSV files.

    Side Effects:
        - Creates a directory for the site if it doesn't already exist.
        - Saves the input dataframes to CSV files in the site's directory.
        - Outputs confirmation messages to indicate that files have been created.
    """

    # Create variable for config to enable f-string
    proc_file_path = CONFIG["PROCESSED_FILE_PATH"]   
    output_directory = rf"{proc_file_path}\{site_name}"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save daytime_model_df to csv (pcorrA)
    decamin_output_file = rf'{output_directory}\{site_name}_decamin_{deployment_duration_str}.csv'
    daytime_model_df.to_csv(decamin_output_file)
    print(f"Created {decamin_output_file} file.")

    # Save daily_df to csv (clear_stats)
    daily_output_file = rf'{output_directory}\{site_name}_daily_{deployment_duration_str}.csv'
    daily_df.to_csv(daily_output_file)
    print(f"Created {daily_output_file} file.")

    # Save cloudless csv file
    cloudless_output_file = rf'{output_directory}\{site_name}_cloudless_{deployment_duration_str}.csv'
    cloudless_df.to_csv(cloudless_output_file)
    print(f"Created {cloudless_output_file} file.")

    # Save filtered cloudless csv file
    filtered_cloudless_output_file = rf'{output_directory}\{site_name}_filtered_cloudless_{deployment_duration_str}.csv'
    filtered_cloudless_df.to_csv(filtered_cloudless_output_file)
    print(f"Created {filtered_cloudless_output_file} file.")


# %%
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the file to process."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with the provided file path
    main(args.input_file)