"""
Main file for AIMS historical weather station surface PAR dataset corrections.

Author: tarmstro
"""

# imports
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore
import math
from datetime import timedelta, date
from statistics import StatisticsError
import par_algos
import par_plots
import himawari_interpolation
import os
from datetime import datetime

# config
from config import CONFIG

# dev only
from docstring_wrapper_dev import log_args_and_types

# TODO: Adjust main() so it is less of a mess
# TODO: Adjust the file intake - using config for the input file path is clunky - consider a basic file selection gui?
# TODO: Better visualisations - Assess if it is worth using plotly for better interactivity of plots (par_plots)
# TODO: Validation of outputs?
# TODO: Continue with/finish docstrings.
# TODO: Write an appropriate README.
# TODO: Should solar noon be calculated with raw or model par?
# TODO: Can we screen shadows out of daily data - Look at the ratio of raw par to model par to identify shadows according to some cutoff?
# TODO: Check the new csv for Cloudless days with outlier days removed - Make the output section neater regarding this
# TODO: Is the pratio correction based on the entire deployment? Does missing data in a deployment cause issues here?

def main():
    """
    
    Main script - Runs all other functions. 

    """   

    script_start_time = datetime.now()
    df, site_name = data_setup()
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
    Get himawari data from the nci thredds server. 
    Data is hourly and will be linearly interpolated to fill data gaps.
    Interpolation of the data is performed by Inverse-Distance-Weighted Interpolation using KDTree as described in indisttree.py
    Himawari data on the thredds server starts on 01/04/2019.

    Args:
        df (DataFrame): Dataframe containing PAR data, latitude, longitude, times.
        site_name (string): _description_
        daytime_df (DataFrame): Dataframe containing PAR data etc which has been filtered to only include daytime values.
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
    Linearly interpolate all himawari data to 10 minute intervals from the hourly interval on the thredds server.
    This is done using the resample() function

    Args:
        data (DataFrame): Dataframe containing PAR data etc along with hourly himawari data.

    Returns:
        df_interpolated (DataFrame): Dataframe containing the now resampled himawari data along with PAR etc.
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


def data_setup():
    """Initial basic wrangling of raw PAR data and associated metadata. Adjusting types, timezone etc.

    Returns:
        data(Dataframe): Dataframe containing all raw PAR data and associated metadata.
    """
    print("Setting up data..")
    df = pd.read_csv(CONFIG["INPUT_FILE"])
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
    """Further basic data wrangling, generating zenith metadata, deployment start/end etc.

    Args:
        data (DataFrame): Dataframe containing raw data with some wrangling applied

    Returns:
        day_time_df (Dataframe): Dataframe containing only daytime PAR values and associated metadata.
        deployment_start_dates (list): List of deployment start dates.
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
    """Get the start dates of all deployments in the dataset.

    Args:
        day_time_df (Dataframe): Dataframe containing PAR data filtered to exclude night time.

    Returns:
        deployment_start_dates (list): List of deployment start dates.
    """
    deployment_start_dates = []
    for i in range(len(day_time_df)):
        if day_time_df['dn1'].iloc[i] == 0:
            deployment_start_dates.append(day_time_df[date].iloc[0])
            print(f"Start of a deployment noted at {day_time_df[date].iloc[0]}.")

    return deployment_start_dates


def get_model_corr_par(daytime_model_df):
    """Build a model of the expected PAR values and get the old corrected PAR values ("old" method) 

    Args:
        daytime_model_df (DataFrame): Dataframe containing both raw and model PAR data filtered to exclude night time.

    Returns:
        daytime_model_df_updated (Dataframe): Dataframe containing PAR data filtered to exclude night time with model and corrected par values appended.
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
    """Assess how far the apparent maximum par is from the model to calculate tilt with a midday filter.

    Args:
        df (DataFrame): Dataframe of all PAR data and associated metadata

    Returns:
        delineation_vals_list (list): List of how many 10 minute periods the maximum raw PAR value differs from the maximum model PAR values (used as approx tilt).
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
            # print("No midday values exist to calculate tilt on: ", current_date, " - Set tilt to 0")
            delineation_val = 0
            delineation_vals_list.append(delineation_val)
            continue

    # print(delineation_vals_list)
    # filtered_delineation_df = pd.DataFrame(delineation_vals_list, columns=['filtered_tilt'])
    # print(filtered_delineation_df)
    # filtered_delineation_df['abs_tilt'] = np.abs(filtered_delineation_df['filtered_tilt'])
    # print("del df: ", len(filtered_delineation_df))
    return delineation_vals_list


def get_raw_tilt(df):
    """Assess how far the apparent maximum par is from the model to calculate tilt. 

    Args:
        df (DataFrame): Dataframe of all PAR data and associated metadata

    Returns:
        raw_delineation_vals_list (list): List of how many 10 minute periods the maximum raw PAR value differs from the maximum model PAR values (used as approx tilt).
    """
    raw_delineation_vals_list = []
    for current_date, day in df.groupby(df['date'].dt.date):
        # Tilt values without middle of the day restriction
        raw_delineation_val = np.argmax(day['modpar']) - np.argmax(day['rawpar'])
        raw_delineation_vals_list.append(raw_delineation_val)
    return raw_delineation_vals_list


def build_daily_cloudless_df(daytime_model_df, raw_tilt, filtered_tilt, abs_tilt):
    """Filters and condenses the wrangled raw dataframe into daily values of daytime data and then further filters to exclude cloudy days

    Args:
        df (DataFrame): _description_
        raw_tilt (list): Raw tilt list
        filtered_tilt (list): Filtered tilt list
        abs_tilt (ndarray): Absolute tilt array

    Returns:
        daily_df (Dataframe): Dataframe containing data condensed down to daily values of daytime data
        cloudless_df (Dataframe): Dataframe containing data condensed down to daily values of daytime data filtered to only include cloudless days
    """

    print("Building daily and cloudless dataframes..")
    # current_instrument = df['instrument_serial_no'].iloc[0]
    daily_df_list = []
    daily_df_cloudless = []
    cloudless_list = []
    ratiop_list = []
    cloudless_flag_list = [0] * len(daytime_model_df)
    daytime_model_df['cloudless_flag'] = cloudless_flag_list
    cloudless_dates = []
    counter = 0

    for current_date, day in daytime_model_df.groupby(daytime_model_df['date'].dt.date):
        try:
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
    """Added functionality of rolling tilt count.

    Functionality enables us to see where instruments may have fallen or otherwise moved for extended periods.

    Args:
        daily_df (Dataframe): Dataframe of daily PAR values.
        cloudless_df (Dataframe): Dataframe of daily PAR values excluding all cloudless days.

    Returns:
        daily_tilt_df (Dataframe): Dataframe of daily PAR values with consecutive tilt values added.
        cloudless_tilt_df (Dataframe): Dataframe of daily PAR values excluding all cloudless days with consecutive tilt values added.
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
    """Adding count of cloudless days and recording where the max_count is exceeded.

    Args:
        daily_df (Dataframe): Dataframe of daily PAR values.

    Returns:
        consec_cloudless_df (Dataframe): Dataframe of daily PAR values with added column tracking consecutive cloudless days with a 1 or 0 for exceeding the max_count or not.
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
    """Calculation of corrected PAR coefficients.

    Polynomial fit test run with specified degrees of freedom. Outliers are then filtered out based on specified standard deviation.
    Polynomial fit test then run again on filtered data. 
    A linear fit test is also run with a comparison printed in the console. 

    Using the coefficients from each test the best fit is established and these coefficients are then used for the correction.

    Args:
        cloudless_df (Dataframe): Dataframe of daily PAR values excluding cloudy days.

    Returns:
        cloudless_df (Dataframe): Dataframe of daily PAR values excluding cloudy days with corrected PAR values.
        used_coeffs (List): List of the coefficients used from the polynomial test.
    """
    # Correcting PAR
    # Create empy lists
    corrected_ratio_list = []
    corrected_par_list = []
    # cloudless_df = cloudless_df[(cloudless_df['ratio_noon_par'] > 2)]
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
        # corrected_ratio = coeffs_filt[0] * y_filt.iloc[i] + coeffs_filt[1] * zenith_correction
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
    """Run through main functions to get tilt, clear stats, cloudless days, par coefficients and finally corrected par values.

    Args:
        deployment_start_dates (list): List of deployment start dates.
        daytime_df (Dataframe): Dataframe of PAR data excluding night time values.
        site_name (string): Site name string.

    Returns:
        daytime_model_df (Dataframe): Final dataframe for daytime par values (10 minute samples).
        daily_df (Dataframe): Final dataframe for daily par values (daily samples).
        cloudless_df (Dataframe): Final dataframe for cloudless day par values (daily samples).
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
        single_deployment_daytime_df = get_model_corr_par(single_deployment_daytime_df)

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
        if len(cloudless_df) < CONFIG["MINIMUM_CLOUDLESS_DAYS"]:
            print(f"Less than {CONFIG["MINIMUM_CLOUDLESS_DAYS"]} cloudless days in this deployment, moving to next deployment.")
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
    """Saves the par outputs as csv files and build plots.

    Args:
        final_daytime_model_df (Dataframe): Final dataframe for daytime par values (10 minute samples).
        final_daily_df (Dataframe): Final dataframe for daily par values (daily samples).
        final_cloudless_df (Dataframe): Final dataframe for cloudless day par values (daily samples).
        deployment_start_dates (list): List of deployment start dates.
        site_name_slice (string): Site name string.
    """

    output_directory = rf"{CONFIG["PROCESSED_FILE_PATH"]}\{site_name}"

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
    main()