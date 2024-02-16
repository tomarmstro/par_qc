# dictionary of configuration options, or maybe should be frozen dataclass?
CONFIG = {}


## PAR Corrections
# constants
# Assessing tilt value only around noon (11-1)
CONFIG["TILT_START_TIME"] = 11
CONFIG["TILT_END_TIME"] = 13
CONFIG["MINIMUM_CLOUDLESS_DAYS"] = 5
CONFIG["CLOUDLESS_THRESHOLD"] = 0.1

# paths
CONFIG["RATIOS_SCATTER_PLOT_FILENAME"] = r'C:\Users\tarmstro\Projects\par_qc\processed\ratios.png'
CONFIG["TILT_PLOT_FILENAME"] = r'C:\Users\tarmstro\Projects\par_qc\processed\tilt.png'
CONFIG["PAR_PLOT_FILENAME"] = r'C:\Users\tarmstro\Projects\par_qc\processed\par.png'
CONFIG["INPUT_FILE"] = r'C:\Users\tarmstro\Projects\par_qc\data\weather_station_datasets\davies_par_export_reduced2.csv'
CONFIG['DAILY_OUTPUT_PATH'] = r"C:\Users\tarmstro\Projects\par_qc\processed\daily_"
CONFIG['CLEARSTATS_OUTPUT_PATH'] = r"C:\Users\tarmstro\Projects\par_qc\processed\clearstats_"
CONFIG['CLOUDLESS_OUTPUT_PATH'] = r"C:\Users\tarmstro\Projects\par_qc\processed\cloudless_"
CONFIG['FULL_OUTPUT_PATH'] = rf'C:\Users\tarmstro\Projects\par_qc\processed\daily_full_'
CONFIG['CLEARSTATS_FULL_OUTPUT_PATH'] = rf'C:\Users\tarmstro\Projects\par_qc\processed\clear_stats_full_'
CONFIG['CLOUDLESS_FULL_OUTPUT_PATH'] = rf'C:\Users\tarmstro\Projects\par_qc\processed\cloudless_full_'


## Interpolation
CONFIG['CSV_FILENAME'] = 'davies_hourly_test_19th.csv'
# CONFIG['DATA_INTERVAL'] = 'hourly'
CONFIG['DATA_INTERVAL'] = 'daily'
CONFIG['FILTER_DEGREES'] = 0.1

# Davies
CONFIG['TARGET_LATITUDE'] = -18.83162
CONFIG['TARGET_LONGITUDE'] = 147.6345

# Thurs
# CONFIG['TARGET_LATITUDE'] = -10.555291
# CONFIG['TARGET_LONGITUDE'] = 142.253283

# START_YEAR = 2019
# START_YEAR_MONTH = 4
CONFIG['START_YEAR'] = 2022
CONFIG['START_YEAR_MONTH'] = 12

# END_YEAR = 2023
CONFIG['END_YEAR'] = 2022
# END_YEAR_MONTH = 9
CONFIG['END_YEAR_MONTH'] = 12