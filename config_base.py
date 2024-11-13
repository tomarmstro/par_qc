""" 
Dictionary of configuration options - Maybe this should be frozen dataclass?
"""
# Initialise dictionary
CONFIG = {}

# Directory/file paths

CONFIG["BASE_FILE_PATH"] = r'C:\Users\tarmstro\Projects'
CONFIG["PROCESSED_FILE_PATH"] = r'C:\Users\tarmstro\Projects\par_qc_ta\processed'
CONFIG["INPUT_FILE"] = r'C:\Users\tarmstro\Projects\par_qc_ta\data\Cleveland_Bay_PAR_2012_2024.csv'


# PAR corrections constants
CONFIG["MINIMUM_CLOUDLESS_DAYS"] = 5
CONFIG["CLOUDLESS_THRESHOLD"] = 0.1
# Assessing tilt value only around noon (11-1)
CONFIG["TILT_START_TIME"] = 11
CONFIG["TILT_END_TIME"] = 13
CONFIG['FIRST_HIMAWARI_DATA_DATE'] = r"2019-04-01"
CONFIG['FILTER_DEGREES'] = 0.1
CONFIG['RATIO_THRESHOLD'] = 1.5

## Thredds url
# CONFIG['THREDDS_URL'] = 'https://thredds.nci.org.au/thredds/catalog/rv74/satellite-products/arc/der/himawari-ahi/solar/p1h/latest/catalog.html'
CONFIG['THREDDS_URL'] = 'https://dapds00.nci.org.au/thredds/catalog/rv74/satellite-products/arc/der/himawari-ahi/solar/p1h/latest/catalog.html'

# Thredds data interval
CONFIG['DATA_INTERVAL'] = 'hourly'
# CONFIG['DATA_INTERVAL'] = 'daily'