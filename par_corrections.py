# imports
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import stdev
from scipy.stats import zscore
import math
from datetime import timedelta, date
from statistics import StatisticsError


## assessing tilt value only around noon (11-1)
TILT_START_TIME = 11
TILT_END_TIME = 13
RATIOS_SCATTER_PLOT_FILENAME = r'C:\Users\tarmstro\Python\par_qc\processed\ratios.png'
TILT_PLOT_FILENAME = r'C:\Users\tarmstro\Python\par_qc\processed\tilt.png'
PAR_PLOT_FILENAME = r'C:\Users\tarmstro\Python\par_qc\processed\par.png'

# Run pcorrC() to run whole script
# pcorrC()

# config
input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\data\weather_station_datasets\davies_17-21.csv'
# input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\data\weather_station_datasets\davies_21-23.csv'
# input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\data\weather_station_datasets\lizard_20-21.csv'
# input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\data\weather_station_datasets\thurs_17_20.csv'
# input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\data\weather_station_datasets\thurs_20-23.csv'
# input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\data\weather_station_datasets\test.csv'

# input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\data\thurs_par_export.csv'
pcorrA_output_file = r'C:\Users\tarmstro\Python\par_qc\processed\python_output_corrA.csv'
clear_stats_output_file = r'C:\Users\tarmstro\Python\par_qc\processed\clear_stats.csv'
cloudless_output_file = r'C:\Users\tarmstro\Python\par_qc\processed\cloudless.csv'

# dav1
# lat0 = -18.8316 * math.pi / 180.0
# long0 = 147.6345


def main_func():
    df = data_setup()
    daytime_df = daytime_dataset_setup(df)
    modpar_df = get_model_corr_par(daytime_df)
    pcorrB(modpar_df)


# Zenith angle
def zen(lat0, long0, dn, hr0, min0):
    # xl is a yearly time scale extending from 0 to 2 pi.
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
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['time']).dt.tz_convert('Etc/GMT-10')
    df['date'] = df['date'] - timedelta(minutes=10)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
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
        dn = date(day_time_df['year'].iloc[i], day_time_df['month'].iloc[i], day_time_df['day'].iloc[i]).timetuple().tm_yday
        dn_list.append(dn)
        z = zen(day_time_df['latitude'].iloc[0] * math.pi / 180.0, day_time_df['longitude'].iloc[0], dn, day_time_df['hour'].iloc[i], day_time_df['minute'].iloc[i])
        z_func_list.append(z)
        del1 = day_time_df['year'].iloc[i] - day_time_df['year'].iloc[0]
        # Dayseq algorithm
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
    day_time_df = day_time_df[['date', 'dn', 'dn1', 'day', 'month', 'year',
                               'hour', 'minute', 'zenith_angle', 'radius_vec', 'equ_time', 'declination', 'rawpar']]

    # Filter by zenith
    day_time_df = day_time_df[(day_time_df["zenith_angle"] > 0) & (day_time_df["zenith_angle"] < 90)].copy()
    return day_time_df

def get_model_corr_par(modpar_df):
    print("Calculating Model PAR..")
    print("Calculating Corrected PAR..")

    corrparr_list = []
    modpar_list = []
    # Get model_par
    for i in range(len(modpar_df)):
        modpar = get_model_par(modpar_df['zenith_angle'].iloc[i])
        # Correct for sun-earth distance (radius vector)
        modpar = modpar * modpar_df['radius_vec'].iloc[i]  # Placeholder for rv value
        modpar_list.append(modpar)

        # Should this 'dn1/dn' value be the days into deployment??
        # pratio = 0.0001221 * modpar_df['dn1'].iloc[i] + 0.95767
        pratio = 0.0001221 * modpar_df['dn'].iloc[i] + 0.95767

        corrpar = modpar_df['rawpar'].iloc[i] * pratio
        corrparr_list.append(corrpar)
    modpar_df['modpar'] = modpar_list
    modpar_df['corrpar'] = corrparr_list
    modpar_df.to_csv(pcorrA_output_file)
    print("Created ", pcorrA_output_file)
    return modpar_df



def pcorrB(df):
    print("Running pcorrB..")

    clear_stats_list = []
    clear_stats_cloudless = []
    cloudless_list = []
    ratiop_list = []

    cloudless_flag_list = [0] * len(df)
    df['cloudless_flag'] = cloudless_flag_list
    cloudless_dates = []
    for date, day in df.groupby(df['date'].dt.date):
        # Restricting tilt calculation to the middle of the day (typically 11-13 hrs)
        tilt_df = day.loc[(day['hour'] >= TILT_START_TIME) & (day['hour'] < TILT_END_TIME)].copy()
        midday_rawpar = tilt_df['rawpar']
        midday_modpar = tilt_df['modpar']

        try:
            index_of_daily_max_rawpar = np.argmax(midday_rawpar)
            index_of_daily_max_modpar = np.argmax(midday_modpar)
            # Tilt value = how many 10 minute periods does the max rawpar differ from max modpar
            delineation_val = index_of_daily_max_modpar - index_of_daily_max_rawpar
        # If no noon values exist (ie instrument recovered early that day), set tilt to 0
        except ValueError as error:
            print(error)
            print("No midday values exist to calculate tilt on: ", date, " - Set tilt to 0")
            delineation_val = 0

        # Tilt values without middle of the day restriction
        old_delineation_value = np.argmax(day['modpar']) - np.argmax(day['rawpar'])

        try:
            const, const1 = statsxy(day['rawpar'], day['modpar'])
        except StatisticsError:
            # only one par value left in the day = can't get variance, so skip this day
            ratiop_list.append(np.nan)
            continue

        # Do we call noon at the highest par for raw or model??
        noon_rawpar_index = np.argmax(day['rawpar'])
        noon_modpar_index = np.argmax(day['modpar'])
        noon_rawpar = day['rawpar'].iloc[noon_modpar_index]
        noon_modpar = day['modpar'].iloc[noon_modpar_index]

        sum_rawpar = (600.0 / 10 ** 6) * np.sum(day['rawpar'])
        sum_modpar = (600.0 / 10 ** 6) * np.sum(day['modpar'])

        clear_stats_list.append((day['date'].iloc[0], day['dn1'].iloc[0], day['dn'].iloc[0],
                                 df['day'].iloc[0], day['month'].iloc[0], day['year'].iloc[0]) +
                                tuple(const1[:19]) + (sum_rawpar, sum_modpar, delineation_val, old_delineation_value, noon_rawpar, noon_modpar))
        dx = 0.10
        if (const1[5] <= dx or const1[5] <= dx or const1[6] <= dx or const1[7] <= dx or
                const1[8] <= dx or const1[9] <= dx or const1[10] <= dx or const1[11] <= dx or
                const1[12] <= dx or const1[13] <= dx or const1[14] <= dx or const1[15] <= dx):
            ratio_sum_par = sum_modpar / sum_rawpar
            ratio_noon_par = noon_modpar / noon_rawpar
            cloudless_list.append((day['date'].iloc[0], day['dn1'].iloc[0], day['dn'].iloc[0],
                                   df['day'].iloc[0], day['month'].iloc[0], day['year'].iloc[0],
                                   sum_rawpar, sum_modpar, ratio_sum_par,
                                   noon_rawpar, noon_modpar, ratio_noon_par,
                                   delineation_val, old_delineation_value))
            # ratiop_list.append(ratiop)
            clear_stats_cloudless.append(1)
            cloudless_dates.append(day['date'].iloc[0].date())
        else:
            # ratiop_list.append(np.nan)
            clear_stats_cloudless.append(0)
        # dayold = day0
        # iold = ii
        cloudless_days = []
        if day['date'].iloc[0].date() in cloudless_dates:
            cloudless_days.append(1)
        else:
            cloudless_days.append(0)

    clear_stats_df = pd.DataFrame(clear_stats_list)

    # Set clear stats column names
    clear_stats_df.columns = ['date', 'dn1', 'dn', 'day', 'month', 'year', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6',
                              '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8',
                              '1.9', 'sum_rawpar', 'sum_modpar', 'tilt', 'old_tilt', 'noon_rawpar', 'noon_modpar']
    clear_stats_df['abs_tilt'] = np.abs(clear_stats_df['tilt'])



    # Build cloudless dataframe and set column names
    cloudless_df = pd.DataFrame(cloudless_list)
    cloudless_df.columns = ['date', 'dn1', 'dn', 'day', 'month', 'year', 'sum_rawpar', 'sum_modpar', 'ratio_sum_par',
                            'noon_rawpar', 'noon_modpar', 'ratio_noon_par', 'tilt', 'old_tilt']
    # Add cloudless flags to clear_stats
    clear_stats_df['cloudless'] = clear_stats_cloudless

    # Find consecutive tilt values and flag
    consecutive_tilt_count = 0
    max_consecutive_tilt_count = 4  # Number of consecutive tilt values we check for
    max_consecutive_tilt_list = []
    for value in clear_stats_df['abs_tilt']:
        if value >= 5:  # Tilt value we check for
            consecutive_tilt_count += 1
            if consecutive_tilt_count >= max_consecutive_tilt_count:
                max_consecutive_tilt_list.append(1)
            else:
                max_consecutive_tilt_list.append(0)
        else:
            consecutive_tilt_count = 0
            max_consecutive_tilt_list.append(0)
    clear_stats_df['tilted'] = max_consecutive_tilt_list
    clear_stats_df['tilt_rolling_avg'] = clear_stats_df['tilt'].rolling(5).mean()
    cloudless_df['tilt_rolling_avg'] = cloudless_df['tilt'].rolling(5).mean()

    # Find consecutive cloudless days and flag
    consecutive_cloudless_count = 0
    max_consecutive_cloudless_count = 4  # Number of consecutive cloudless days we check for
    max_consecutive_cloudless_list = []
    for value in clear_stats_df['cloudless']:
        if value >= 1:  # Cloudless or not - 1 for cloudless
            consecutive_cloudless_count += 1
            if consecutive_cloudless_count >= max_consecutive_cloudless_count:
                max_consecutive_cloudless_list.append(1)
            else:
                max_consecutive_cloudless_list.append(0)
        else:
            consecutive_cloudless_count = 0
            max_consecutive_cloudless_list.append(0)
    clear_stats_df['consec_clouds'] = max_consecutive_cloudless_list

    # Build clear_stats csv
    clear_stats_df.to_csv(clear_stats_output_file)
    print("Created ", clear_stats_output_file)
    # df['corrpar'] = df['rawpar'] * (0.0001221 * df['dn1'] + 0.95767)
    # df.to_csv(pcorrA_output_file)
    # print("Created ", pcorrA_output_file)

    sorted_indices = np.argsort(cloudless_df['dn1'])
    # print(sorted_indices)

    # x = cloudless_df['dn1'][sorted_indices]
    # y = cloudless_df['ratiop'][sorted_indices]

    # x = cloudless_df['dn'][sorted_indices]
    # y = cloudless_df['ratiop'][sorted_indices]
    # # print(x - cloudless_df['dn1'])

    # checking for outliers
    v = np.polyfit(cloudless_df['dn'], cloudless_df['ratio_noon_par'], 1)
    residuals = cloudless_df['ratio_noon_par'] - (v[0] + v[1] * cloudless_df['dn'])
    z_scores = zscore(residuals)
    filtered_indices = np.abs(z_scores) <= 1.1

    x_filt = cloudless_df['dn'][filtered_indices]
    y_filt = cloudless_df['ratio_noon_par'][filtered_indices]
    v_filt = np.polyfit(x_filt, y_filt, 1)
    print("v = ", v_filt)

    corrected_ratio_list = []
    corrected_par_list = []
    for i in range(len(cloudless_df['dn'])):
        # Remove outlier values
        if filtered_indices[i] == True:
            corrected_ratio = 0.0001221 * x_filt[i] + 0.95767

        else:
            corrected_ratio = np.nan
        # Use this to ignore outlier removal
        # corrected_ratio = 0.0001221 * cloudless_df['dn'][i] + 0.95767
        corrected_ratio_list.append(corrected_ratio)
        pratio = v[0] * modpar_df['dn1'].iloc[i] + v[1]
        corrected_par = noon_rawpar * pratio
        corrected_par_list.append(corrected_par)

    cloudless_df['corrected_ratio'] = corrected_ratio_list
    cloudless_df['corrected_par'] = corrected_par_list
    cloudless_df.to_csv(cloudless_output_file)
    print("Created ", cloudless_output_file)

    build_plots(df, cloudless_df, clear_stats_df)


def build_plots(df, cloudless_df, clear_stats_df):
    # Plotting ratios
    ratio_scatter, ratio_scatter_ax = plt.subplots(figsize=(12,6))
    ratio_scatter_ax.scatter(cloudless_df['date'], cloudless_df['ratio_sum_par'], label='pratio')
    ratio_scatter_ax.scatter(cloudless_df['date'], cloudless_df['corrected'], label='corrected pratio')
    ratio_scatter_ax.legend()
    ratio_scatter_ax.title.set_text('Pratio for cloudless days')
    ratio_scatter.savefig(RATIOS_SCATTER_PLOT_FILENAME)

    # Plotting Tilt Values
    tilt_plot, tilt_plot_ax = plt.subplots(figsize=(12,6))
    tilt_plot_ax.scatter(clear_stats_df['date'], clear_stats_df['tilt'], label='tilt', s=5)
    tilt_plot_ax.scatter(clear_stats_df['date'], clear_stats_df['tilt_rolling_avg'], label='rolling_avg', s=5)
    tilt_plot_ax.legend()
    tilt_plot_ax.title.set_text('Daily tilt values')
    tilt_plot.savefig(TILT_PLOT_FILENAME)


    # Plots Comparing PAR Values
    par_plot, par_plot_ax = plt.subplots(5, 1, figsize=(24, 12))
    par_plot_ax[0].scatter(df['date'], df['rawpar'], label='raw', s=1)
    par_plot_ax[0].legend()
    par_plot_ax[0].title.set_text('Raw Values')

    par_plot_ax[1].scatter(df['date'], df['modpar'], label='mod', s=1)
    par_plot_ax[1].legend()
    par_plot_ax[1].title.set_text('Mod Values')

    par_plot_ax[2].scatter(df['date'], df['corrpar'], label='corr', s=1)
    par_plot_ax[2].legend()
    par_plot_ax[2].title.set_text('Corr Values')

    par_plot_ax[3].scatter(df['date'], (df['rawpar'] - df['corrpar']), label='raw/corr diff', s=1)
    par_plot_ax[3].legend()
    par_plot_ax[3].title.set_text('Difference between Raw and Corrected PAR')

    par_plot_ax[4].scatter(df['date'], (df['rawpar'] - df['modpar']), label='raw/mod diff', s=1)
    par_plot_ax[4].legend()
    par_plot_ax[4].title.set_text('Difference between Raw and Model PAR')

    par_plot.tight_layout(pad=5.0)
    par_plot.savefig(PAR_PLOT_FILENAME)

    print("Created plots.")


# pcorrB(get_model_corr_par())
main_func()