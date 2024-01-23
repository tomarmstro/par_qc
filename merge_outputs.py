import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# config
input_file_pcorrA = r"C:\Users\tarmstro\Python\par_qc\processed\davies_proc\pcorrA_Davies_2021-06-24_2023-07-30.csv"
cloudless_file = r"C:\Users\tarmstro\Python\par_qc\processed\davies_proc\cloudless_Davies_2021-06-24_2023-07-30.csv"
# input_file_pcorrA = r"C:\Users\tarmstro\Python\par_qc\pcorrA_dec2022.csv"
input_file_interpol = r"C:\Users\tarmstro\Python\par_qc\davies_hourly_test_19th.csv"

# Import csv files
df_pcorrA = pd.read_csv(input_file_pcorrA, parse_dates=['date'])
df_cloudless = pd.read_csv(cloudless_file, parse_dates=['date'])
df_interpol = pd.read_csv(input_file_interpol, parse_dates=['date'])

# Concatenate csv files together
# df_all = pd.concat([df_pcorrA, df_interpol], ignore_index = True)
df_all = pd.merge(df_pcorrA, df_interpol, on = 'date')


# Sort by dates and filter to only keep relevant dates
df_all.sort_values(by='date', inplace = True)
df_all = df_all[(df_all['date'] >= df_interpol.iloc[0]['date']) & (df_all['date'] <= df_interpol.iloc[-1]['date'])]

df_all.groupby('date').agg({'date': 'first', 'rawpar': 'first', 'corpar': 'first', 'interpolated_value (umol m-2 s-1)': 'first'}).reset_index(drop=True)
df_noon = df_all.copy()
df_noon['date'] = pd.to_datetime(df_noon['date'])
df_noon.set_index('date', inplace=True)
df_noon = df_noon.resample('D').apply(lambda x: x.between_time('12:30', '12:30'))


# Plots Comparing PAR Values
corrected_vs_interpolated_plot, corrected_vs_interpolated_plot_ax = plt.subplots(1, 1, figsize=(24, 12))
corrected_vs_interpolated_plot_ax.scatter(df_all['date'], df_all['interpolated_value (umol m-2 s-1)'], label='interpolated_value')
corrected_vs_interpolated_plot_ax.scatter(df_all['date'], df_all['corpar'], label='corpar')
corrected_vs_interpolated_plot_ax.legend()
corrected_vs_interpolated_plot_ax.title.set_text('Interpolated Values')
# interpol_plot.tight_layout(pad=5.0)
corrected_vs_interpolated_plot.savefig('corrected_vs_interpolated_plot.png')
#
# # Plots Comparing PAR Values
raw_vs_interpolated_plot, raw_vs_interpolated_plot_ax = plt.subplots(1, 1, figsize=(24, 12))
raw_vs_interpolated_plot_ax.scatter(df_all['date'], df_all['interpolated_value (umol m-2 s-1)'], label='interpolated_value')
raw_vs_interpolated_plot_ax.scatter(df_all['date'], df_all['rawpar'], label='rawpar')
raw_vs_interpolated_plot_ax.legend()
raw_vs_interpolated_plot_ax.title.set_text('Interpolated Values')
# interpol_plot.tight_layout(pad=5.0)
raw_vs_interpolated_plot.savefig('raw_vs_interpolated_plot.png')

# Export to csv
df_all.to_csv('merged_data.csv')
df_noon.to_csv('merged_data_noon.csv')
# df_cloudless.to_csv('cloudless_data_noon.csv')

