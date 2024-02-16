"""
PAR Corrections
Author: tarmstro
"""

# imports
from sklearn.linear_model import LinearRegression
# import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import stdev
from scipy.stats import zscore
import math
from datetime import timedelta, date
from statistics import StatisticsError

#config
from config import CONFIG

def main():
    df = data_setup()
    site_name = df['site_name'][0]
    print(f"Processing data from {site_name}..")
    site_name_slice = site_name[0:6]
    daytime_df, deployment_start_dates = daytime_dataset_setup(df)

    blank = []
    final_daytime_model_df = pd.DataFrame(blank)
    final_daily_df = pd.DataFrame(blank)
    final_cloudless_df = pd.DataFrame(blank)
    for i in range(len(deployment_start_dates)-1):
        print("\n***************** New deployment *****************")
        deployment_start_date = str(deployment_start_dates[i])
        print("Deployment Start: ", deployment_start_date)
        deployment_end_date = str(deployment_start_dates[i+1])
        print("Deployment End: ", deployment_end_date)
        mask = (daytime_df['date'] > deployment_start_date) & (daytime_df['date'] <= deployment_end_date)
        masked_data = daytime_df.loc[mask]
        daytime_model_df = get_model_corr_par(masked_data)

        # Get tilt values
        old_tilt = get_raw_tilt(daytime_model_df)
        filtered_tilt = get_filtered_tilt(daytime_model_df)
        abs_tilt = np.abs(filtered_tilt)

        # Get clear_stats and cloudless dfs
        daily_df, cloudless_df = build_daily_cloudless_df(daytime_model_df, old_tilt, filtered_tilt, abs_tilt)

        # Get tilt and cloudless flags - add to dfs
        clear_stats_tilt_df, cloudless_tilt_df = get_consecutive_tilt(daily_df, cloudless_df)
        daily_df = daily_df.join(clear_stats_tilt_df)
        cloudless_df = cloudless_df.join(cloudless_tilt_df)

        if len(cloudless_df) < CONFIG["MINIMUM_CLOUDLESS_DAYS"]:
            print(f"Less than {CONFIG["MINIMUM_CLOUDLESS_DAYS"]} cloudless days in this deployment, moving to next deployment.")
            continue
        daily_df = daily_df.join(get_consecutive_cloudless(daily_df))

        # Get corrected par
        print("Calculating corrected PAR..")
        cloudless_df, coeffs = correct_par(cloudless_df)

        daytime_model_df['corpar'] = (coeffs[0] * daytime_model_df['dn1'] + coeffs[1]) * daytime_model_df['rawpar']
        daytime_model_df['coeff1'] = coeffs[0]
        daytime_model_df['coeff2'] = coeffs[1]

        daily_df['noon_corpar'] = (coeffs[0] * daily_df['dn1'] + coeffs[1]) * daily_df['noon_rawpar']
        daily_df['coeff1'] = coeffs[0]
        daily_df['coeff2'] = coeffs[1]

        daytime_model_df.to_csv(CONFIG['DAILY_OUTPUT_PATH'] +
                                site_name_slice + "_" + deployment_start_date + "_" + deployment_end_date + ".csv")
        final_daytime_model_df = pd.concat([final_daytime_model_df, daytime_model_df], ignore_index=True)
        daily_df.to_csv(CONFIG['CLEARSTATS_OUTPUT_PATH'] +
                        site_name_slice + "_" + deployment_start_date + "_" + deployment_end_date + ".csv")
        final_daily_df = pd.concat([final_daily_df, daily_df], ignore_index=True)
        cloudless_df.to_csv(CONFIG['CLOUDLESS_OUTPUT_PATH'] +
                            site_name_slice + "_" + deployment_start_date + "_" + deployment_end_date + ".csv")
        final_cloudless_df = pd.concat([final_cloudless_df, cloudless_df], ignore_index=True)
    # Save daytime_model_df to csv (pcorrA)
    pcorrA_output_file = CONFIG['FULL_OUTPUT_PATH'] + f'{site_name_slice}.csv'
    final_daytime_model_df.to_csv(pcorrA_output_file)
    print(f"Created {pcorrA_output_file} file.")

    # Save daily_df to csv (clear_stats)
    clear_stats_output_file = CONFIG['CLEARSTATS_FULL_OUTPUT_PATH'] + f'{site_name_slice}.csv'
    final_daily_df.to_csv(clear_stats_output_file)
    print(f"Created {clear_stats_output_file} file.")

    # Save cloudless csv file
    cloudless_output_file = CONFIG['CLOUDLESS_FULL_OUTPUT_PATH'] + f'{site_name_slice}.csv'
    final_cloudless_df.to_csv(cloudless_output_file)
    print(f"Created {cloudless_output_file} file.")

    # Build plots
    build_plots(final_daytime_model_df, final_cloudless_df, final_daily_df, deployment_start_dates)
    print("Created plots.")


# Zenith angle
def zen(lat0, long0, dn, hr0, min0):
    # xl is a yearly timescale extending from 0 to 2 pi.
    xl = 2.0 * math.pi * (dn - 1) / 365.0

    dec = (
            0.006918 - 0.399912 * np.cos(xl) + 0.07257 * np.sin(xl) -
            0.006758 * np.cos(2.0 * xl) + 0.000907 * np.sin(2.0 * xl) -
            0.002697 * np.cos(3.0 * xl) + 0.00148 * np.sin(3.0 * xl)
    )

    rv = (
            1.000110 + 0.034221 * np.cos(xl) + 0.001280 * np.sin(xl) +
            0.000719 * np.cos(2.0 * xl) + 0.000077 * np.sin(2.0 * xl)
    )

    et = (
             0.000075 + 0.001868 * np.cos(xl) - 0.032077 * np.sin(xl) -
             0.014615 * np.cos(2.0 * xl) - 0.04089 * np.sin(2.0 * xl)
     ) * 229.18 / 60.0

    # zenith angle estimated (z)
    tst = hr0 + ((min0 + 5.0) / 60.0) - (4.0 / 60.0) * np.abs(150.0 - long0) + et
    hrang = (12.0 - tst) * 15.0 * math.pi / 180.0
    # Cosine zenith angle
    cz = np.sin(lat0) * np.sin(dec) + np.cos(lat0) * np.cos(dec) * np.cos(hrang)
    # zenith angle
    z = (180.0 / math.pi) * np.arccos(cz)

    # z = zenith angle
    # rv = raduis vector
    # et = equation of time
    # dec = declination
    return z, rv, et, dec


# Get Model PAR value from zenith angle
def get_model_par(z):
    pi = 3.1415926
    z0 = z * pi / 180.0
    # Get cosine of zenith angle
    cz = np.cos(z0)
    modpar = -7.1165 + 768.894 * cz + 4023.167 * cz ** 2 - 4180.1969 * cz ** 3 + 1575.0067 * cz ** 4
    return modpar


def statsxy(x, y):
    const = []
    const1 = []
    for j in range(19):
        dely = 2.0 - j * 0.1
        y1 = y * dely
        const.append(stdev(x - y1))
        const1.append(stdev(x - y1) / np.max(y))
    # np.std gives slightly different constant values
        # const.append(np.std(x - y1))
        # const1.append(np.std(x - y1) / np.max(y))
    return const, const1


def data_setup():
    print("Setting up data..")
    df = pd.read_csv(CONFIG["INPUT_FILE"])
    df['date'] = pd.to_datetime(df['time']).dt.tz_convert('Etc/GMT-10')
    df['date'] = df['date'] - timedelta(minutes=10)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['instrument_serial_no'] = df['serial_num']
    df = df.rename(columns={"raw_value": "rawpar"})
    df.sort_values(by='date', inplace=True)
    data = df.copy()
    return data


def daytime_dataset_setup(data):
    print("Extracting daytime data..")

    # Filter by time
    day_time_df = data.loc[(data['hour'] >= 5.0) | (data['hour'] <= 19.0)].copy()

    # Replaces daynumber function
    dnstart = date(day_time_df['year'].iloc[0], day_time_df['month'].iloc[0],
                   day_time_df['day'].iloc[0]).timetuple().tm_yday

    # Initiate lists for zenith and model_par
    z_func_list = []
    dn1_list = []
    dn_list = []

    # Get zenith from zen()
    for i in range(len(day_time_df)):
        # zenith angle z estimated. if z lt 0 skip
        dn = date(day_time_df['year'].iloc[i], day_time_df['month'].iloc[i],
                  day_time_df['day'].iloc[i]).timetuple().tm_yday
        dn_list.append(dn)
        z = zen(day_time_df['latitude'].iloc[0] * math.pi / 180.0, day_time_df['longitude'].iloc[0],
                dn, day_time_df['hour'].iloc[i], day_time_df['minute'].iloc[i])
        z_func_list.append(z)
        del1 = day_time_df['year'].iloc[i] - day_time_df['year'].iloc[0]
        # Dayseq algorithm
        # Deprecated - reassigned in later function
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

    # Filter by zenith
    day_time_df = day_time_df[(day_time_df["zenith_angle"] > 0) & (day_time_df["zenith_angle"] < 90)].copy()
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
    deployment_start_dates = []
    for i in range(len(day_time_df)):
        if day_time_df['dn1'].iloc[i] == 0:
            deployment_start_dates.append(day_time_df[date].iloc[0])
            print(f"Start of a deployment noted at {day_time_df[date].iloc[0]}.")

    return deployment_start_dates


def get_model_corr_par(daytime_model_df):
    print("Calculating model PAR..")
    

    old_corpar_list = []
    modpar_list = []
    dn1_update_list = []
    dn1_update_counter = 1
    # Get model_par
    for i in range(len(daytime_model_df)):
        modpar = get_model_par(daytime_model_df['zenith_angle'].iloc[i])
        # Correct for sun-earth distance (radius vector)
        modpar = modpar * daytime_model_df['radius_vec'].iloc[i]  # Placeholder for rv value
        modpar_list.append(modpar)

        # Should this 'dn1/dn' value be the days into deployment??
        # pratio = 0.0001221 * daytime_model_df['dn1'].iloc[i] + 0.95767
        pratio = 0.0001221 * daytime_model_df['dn1'].iloc[i] + 0.95767

        old_corpar = daytime_model_df['rawpar'].iloc[i] * pratio
        old_corpar_list.append(old_corpar)
        dn1_update_list.append(daytime_model_df['dn1'].iloc[i]-daytime_model_df['dn1'].iloc[0]+1)
        dn1_update_counter += 1
        # daytime_model_df['dn1'] = daytime_model_df['dn1']-daytime_model_df['dn1'].iloc[0]
    daytime_model_df_copy = daytime_model_df.copy()
    daytime_model_df_copy['dn1'] = dn1_update_list
    daytime_model_df_copy['modpar'] = modpar_list
    daytime_model_df_copy['old_corpar'] = old_corpar_list

    return daytime_model_df_copy


def get_filtered_tilt(df):
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
    old_delineation_vals_list = []
    for current_date, day in df.groupby(df['date'].dt.date):
        # Tilt values without middle of the day restriction
        old_delineation_val = np.argmax(day['modpar']) - np.argmax(day['rawpar'])
        old_delineation_vals_list.append(old_delineation_val)
    return old_delineation_vals_list


def build_daily_cloudless_df(df, old_tilt, filtered_tilt, abs_tilt):
    print("Building daily and cloudless dataframes..")
    # current_instrument = df['instrument_serial_no'].iloc[0]
    daily_df_list = []
    daily_df_cloudless = []
    cloudless_list = []
    ratiop_list = []

    cloudless_flag_list = [0] * len(df)
    df['cloudless_flag'] = cloudless_flag_list
    cloudless_dates = []
    # for date, day in df.groupby(df['date'].dt.date):
    counter = 0
    # dn1_counter = 0
    for current_date, day in df.groupby(df['date'].dt.date):
        try:
            const, const1 = statsxy(day['rawpar'], day['modpar'])
        except StatisticsError:
            # only one par value left in the day = can't get variance, so skip this day
            ratiop_list.append(np.nan)
            continue

        # day.loc[:, ('dn1', 0)] = dn1_counter
        #
        # if day['instrument_serial_no'].iloc[0] != current_instrument:
        #     print(f"Instrument changed from {current_instrument} to {day['instrument_serial_no'].iloc[0]} on {date}!")
        #     current_instrument = day['instrument_serial_no'].iloc[0]
        #     dn1_counter = 0
        # else:
        #     dn1_counter += 1

        # Do we call noon at the highest par for raw or model??
        noon_rawpar_index = np.argmax(day['rawpar'])
        noon_modpar_index = np.argmax(day['modpar'])
        noon_rawpar = day['rawpar'].iloc[noon_modpar_index]
        noon_modpar = day['modpar'].iloc[noon_modpar_index]

        sum_rawpar = (600.0 / 10 ** 6) * np.sum(day['rawpar'])
        sum_modpar = (600.0 / 10 ** 6) * np.sum(day['modpar'])

        # Collate all clear stats data
        daily_df_list.append((day['date'].iloc[0], day['dn1'].iloc[0], day['dn'].iloc[0],
                              day['day'].iloc[0], day['month'].iloc[0], day['year'].iloc[0]) +
                             tuple(const1[:19]) + (sum_rawpar, sum_modpar, noon_rawpar, noon_modpar,
                                                   old_tilt[counter], filtered_tilt[counter],
                                                   abs_tilt[counter]))

        # Select for cloudless days
        if (const1[5] <= CONFIG["CLOUDLESS_THRESHOLD"] or const1[5] <= CONFIG["CLOUDLESS_THRESHOLD"] or
                const1[6] <= CONFIG["CLOUDLESS_THRESHOLD"] or const1[7] <= CONFIG["CLOUDLESS_THRESHOLD"] or
                const1[8] <= CONFIG["CLOUDLESS_THRESHOLD"] or const1[9] <= CONFIG["CLOUDLESS_THRESHOLD"] or
                const1[10] <= CONFIG["CLOUDLESS_THRESHOLD"] or const1[11] <= CONFIG["CLOUDLESS_THRESHOLD"] or
                const1[12] <= CONFIG["CLOUDLESS_THRESHOLD"] or const1[13] <= CONFIG["CLOUDLESS_THRESHOLD"] or
                const1[14] <= CONFIG["CLOUDLESS_THRESHOLD"] or const1[15] <= CONFIG["CLOUDLESS_THRESHOLD"]):
            ratio_sum_par = sum_modpar / sum_rawpar
            ratio_noon_par = noon_modpar / noon_rawpar
            cloudless_list.append((day['date'].iloc[0], day['dn1'].iloc[0], day['dn'].iloc[0],
                                   day['day'].iloc[0], day['month'].iloc[0], day['year'].iloc[0],
                                   sum_rawpar, sum_modpar, ratio_sum_par,
                                   noon_rawpar, noon_modpar, ratio_noon_par,
                                   old_tilt[counter], filtered_tilt[counter],
                                   abs_tilt[counter]))
            # ratiop_list.append(ratiop)
            daily_df_cloudless.append(1)
            cloudless_dates.append(day['date'].iloc[0].date())
        else:
            # ratiop_list.append(np.nan)
            daily_df_cloudless.append(0)
        # dayold = day0
        # iold = ii
        cloudless_days = []
        if day['date'].iloc[0].date() in cloudless_dates:
            cloudless_days.append(1)
        else:
            cloudless_days.append(0)
        counter += 1
    daily_df = pd.DataFrame(daily_df_list)

    # Set clear stats column names
    daily_df.columns = ['date', 'dn1', 'dn', 'day', 'month', 'year', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6',
                        '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8',
                        '1.9', 'sum_rawpar', 'sum_modpar', 'noon_rawpar', 'noon_modpar',
                        'old_tilt', 'filtered_tilt', 'abs_tilt']

    # Build cloudless dataframe and set column names
    cloudless_df = pd.DataFrame(cloudless_list)
    cloudless_df.columns = ['date', 'dn1', 'dn', 'day', 'month', 'year', 'sum_rawpar', 'sum_modpar', 'ratio_sum_par',
                            'noon_rawpar', 'noon_modpar', 'ratio_noon_par', 'old_tilt', 'filtered_tilt', 'abs_tilt']
    # Add cloudless flags to clear_stats
    daily_df['cloudless'] = daily_df_cloudless

    return daily_df, cloudless_df


def get_consecutive_tilt(daily_df, cloudless_df):
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
    clear_stats_tilt_df = pd.DataFrame()
    cloudless_tilt_df = pd.DataFrame()
    clear_stats_tilt_df['tilted'] = max_consecutive_tilt_list
    clear_stats_tilt_df['tilt_rolling_avg'] = daily_df['filtered_tilt'].rolling(5).mean()
    cloudless_tilt_df['tilt_rolling_avg'] = cloudless_df['filtered_tilt'].rolling(5).mean()
    return clear_stats_tilt_df, cloudless_tilt_df


def get_consecutive_cloudless(daily_df):
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


def correct_par(cloudless_df):

    # Correcting PAR
    # Create empy lists
    corrected_ratio_list = []
    corrected_par_list = []
    # cloudless_df = cloudless_df[(cloudless_df['ratio_noon_par'] > 2)]
    # Additional cloudless filter (Anything over a certain raw-model par ratio can't be cloudless
    cloudless_df = cloudless_df.drop(cloudless_df[cloudless_df['ratio_noon_par'] >= 1.5].index)
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
    # Get data within standard deviation tolerance/at filterered indicies
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
    else:
        used_coeffs = coeffs_filt[0]

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
    print(f"PAR correction coefficients being used are {used_coeffs[0]} and {used_coeffs[1]}.")
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
    # print(cloudless_df)
    cloudless_df['coeff1'] = used_coeffs[0]
    cloudless_df['coeff2'] = used_coeffs[1]
    return cloudless_df, used_coeffs


def build_plots(df, cloudless_df, daily_df, deployment_start_dates):
    # Plotting ratios
    ratio_scatter, ratio_scatter_ax = plt.subplots(figsize=(12, 6))
    ratio_scatter_ax.scatter(cloudless_df['date'], cloudless_df['ratio_noon_par'], label='ratio_noon_par')
    ratio_scatter_ax.scatter(cloudless_df['date'], cloudless_df['corrected_ratio'], label='corrected_ratio')
    ratio_scatter_ax.legend()
    ratio_scatter_ax.title.set_text('Pratio for cloudless days')
    for date in deployment_start_dates:
        ratio_scatter_ax.axvline(date ,color='r')
    ratio_scatter.savefig(CONFIG["RATIOS_SCATTER_PLOT_FILENAME"])

    # Plotting Tilt Values
    tilt_plot, tilt_plot_ax = plt.subplots(figsize=(12, 6))
    tilt_plot_ax.scatter(daily_df['date'], daily_df['filtered_tilt'], label='filtered_tilt', s=5)
    tilt_plot_ax.scatter(daily_df['date'], daily_df['tilt_rolling_avg'], label='rolling_avg', s=5)
    tilt_plot_ax.legend()
    tilt_plot_ax.title.set_text('Daily tilt values')
    for date in deployment_start_dates:
        tilt_plot_ax.axvline(date ,color='r')
    tilt_plot.savefig(CONFIG["TILT_PLOT_FILENAME"])

    # Plots Comparing PAR Values
    par_plot, par_plot_ax = plt.subplots(5, 1, figsize=(24, 12))
    par_plot_ax[0].scatter(df['date'], df['rawpar'], label='raw', s=1)
    par_plot_ax[0].legend()
    par_plot_ax[0].title.set_text('Raw Values')
    for date in deployment_start_dates:
        par_plot_ax[0].axvline(date ,color='r')

    par_plot_ax[1].scatter(df['date'], df['modpar'], label='mod', s=1)
    par_plot_ax[1].legend()
    par_plot_ax[1].title.set_text('Mod Values')
    for date in deployment_start_dates:
        par_plot_ax[1].axvline(date ,color='r')

    par_plot_ax[2].scatter(df['date'], df['corpar'], label='cor', s=1)
    par_plot_ax[2].legend()
    par_plot_ax[2].title.set_text('Corr Values')
    for date in deployment_start_dates:
        par_plot_ax[2].axvline(date ,color='r')

    par_plot_ax[3].scatter(df['date'], (df['rawpar'] - df['corpar']), label='raw/cor diff', s=1)
    par_plot_ax[3].legend()
    par_plot_ax[3].title.set_text('Difference between Raw and Corrected PAR')
    for date in deployment_start_dates:
        par_plot_ax[3].axvline(date ,color='r')

    par_plot_ax[4].scatter(df['date'], (df['corpar'] - df['modpar']), label= 'cor/mod diff', s=1)
    par_plot_ax[4].legend()
    par_plot_ax[4].title.set_text('Difference between Corrected and Model PAR')
    for date in deployment_start_dates:
        par_plot_ax[4].axvline(date ,color='r')


    par_plot.tight_layout(pad=5.0)
    par_plot.savefig(CONFIG["PAR_PLOT_FILENAME"])


# build_daily_cloudless_df(get_model_corr_par())
# main()
# %%
if __name__ == "__main__":
    main()