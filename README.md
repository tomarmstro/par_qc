
# PAR Data Correction Scripts

This collection of scripts performs **Photosynthetically Active Radiation (PAR)** data corrections on historical weather station data from **AIMS** (Australian Institute of Marine Science). The methodology is based on the research by Manuel et al. (2022), specifically addressing PAR data collected from photovoltaic quantum sensors in remote weather stations on the Great Barrier Reef.

**Reference**:  
*Correcting PAR Data from Photovoltaic Quantum Sensors on Remote Weather Stations on the Great Barrier Reef* (2022)  
**Authors**: M. Nunez, N. Cantin, C. Steinberg, V. Van Dongen-Vogels, S. Bainbridge  
**Journal**: *Journal of Atmospheric and Oceanic Technology*, 39(4):425-448.

---

## Files Overview

### `par_corrections.py`
- **Purpose**: Main processing script for correcting raw PAR data.
- **Functions**:
  - Imports and preprocesses raw PAR data from the AIMS weather station dataset ([AIMS Weather Stations](https://weather.aims.gov.au/#/overview)).
  - Separates data into individual deployments for each instrument.
  - Generates a model of expected PAR values based on zenith angle calculations for the specified latitude.
  - Identifies cloudless days to determine correction coefficients for raw PAR data.
  - Applies correction coefficients to calculate adjusted PAR values.
  - Compares corrected PAR values with sea surface temperature data from the Himawari dataset for accuracy checks.

### `par_plots.py`
- **Purpose**: Generates plots from the outputs of `par_corrections.py` to visualize data corrections and comparisons.
- **Usage**: At the end of the the correction process, `par_plots.py` will run to generate visual outputs.

### `par_algos.py`
- **Purpose**: Contains algorithms specific to PAR data correction, used as helper functions in `par_corrections.py`.
- **Contents**: PAR correction algorithms, solar angle calculations, and other utility functions.

### `himawari_interpolation.py`
- **Purpose**: Ensures required Himawari sea surface temperature (SST) data files are available and interpolates SST values for the relevant geographic coordinates.
- **Functions**:
  - Checks for pre-existing `.csv` files of Himawari data.
  - Downloads missing SST data from the NCI THREDDS server.
  - Interpolates SST values for locations of interest.

### `invdisttree.py`
- **Purpose**: Provides **Inverse-Distance-Weighted (IDW) interpolation** using a **KDTree** for efficient spatial interpolation of Himawari dataset values.
- **Usage**: Called within `himawari_interpolation.py` to handle SST data interpolation.

### `get_thredds_file.py`
- **Purpose**: Retrieves filenames for all hourly Himawari solar dataset files from the NCI THREDDS server.
- **Usage**: Helps manage file access and update checks for Himawari data.
- **Source**: [NCI THREDDS Catalog](https://dapds00.nci.org.au/thredds/catalog/rv74/satellite-products/arc/der/himawari-ahi/solar/p1h/latest/catalog.html)

### `process.py`
- **Purpose**: Orchestrates the execution of the main `par_corrections.py` script, running the full correction process.
- **Usage**: Run this script to start the entire data correction pipeline.

### `config.py`
- **Purpose**: Contains constants, file paths, and adjustable parameters for easy configuration.
- **Contents**: Machine-specific file paths, threshold values for data filtering, and other adjustable settings for adapting the scripts to different environments.

---

## Usage Instructions

1. **Setup**: Ensure all required libraries are installed. Modify `config.py` for your local setup.
2. **Run**: Execute `process.py` to perform the full data correction procedure.
3. **Visualize**: At the end of the the correction process, `par_plots.py` will run to generate visual outputs.

---

## Outputs
**himawari_interpolation/'site'_himawari_results.csv**: A csv file containing the interpolated himawari data for every 10 minute period available. If this doesn't exist on running the script, the nci thredds server will be queried to collect them and store in a csv file. Thredds data collection will take 12+ hours, very fast once collected and stored in .csv file. 
**processed/'site'_decamin.csv**: A csv file containing all variables used for the 10 minute PAR data. 
	Variables/Columns: 
	date: The date the data was collected.
	dn1: Days into deployment.
	dn: Days into the year the data was collected.
	day: The day of the month the data was collected.
	month: The month of the year the data was collected.
	year: The year the data was collected.
	hour: The hour the data was collected.
	minute: The minute the data was collected.
	instrument_serial_no: The serial number of the instrument the data was collected on.
	zenith_angle: The zenith angle of the sun when the data was collected. Calculated in par_algos.py in the zen() function.
	radius_vec: Radius vector, representing Earth's distance from the Sun in astronomical units. Calculated in par_algos.py in the zen() function.
	equ_time: Equation of time in hours, which accounts for Earth's elliptical orbit and axial tilt. Calculated in par_algos.py in the zen() function.
	declination: Solar declination in radians, indicating the angle between Earth's equatorial plane and the Sun's rays. Calculated in par_algos.py in the zen() function.
	rawpar: The raw PAR values directly from the AIMS weather station data webpage.
	interpolated_value (Mj m-2 hr-1): The interpolated himawari sea surface light value in Mj m-2 hr-1. Interpolated using Inverse-Distance-Weighted Interpolation with KDTree.
	interpolated_value (umol m-2 s-1): The interpolated himawari sea surface light value in umol m-2 s-1. Interpolated using Inverse-Distance-Weighted Interpolation with KDTree.
	basic_interpolated_value: The interpolated himawari sea surface light value in Mj m-2 hr-1. Interpolated by simply taking the mean of the nearest 4 geographic points in the himawari dataset.
	himawari_resampled: Himawari data resampled linearly to get values for every 10 minutes instead of once/hour. Unreliable, estimation only.
	modpar: Model par based on zenith angle. 
	old_corpar: Old version of corrected PAR using an superceded algorithm. 
	qc_flag: A binary quality control flag to indicate whether the ratio of raw to model PAR is above the RATIO_THRESHOLD specified in the config.
	cloudless_flag: A binary quality control flag to indicate whether the day is cloudless or not. 
	corpar: The final corrected PAR value.
	coeff1: Coefficient 1 used in the calculation of corrected PAR values.
	coeff2: Coefficient 2 used in the calculation of corrected PAR values.
	
**processed/'site'_daily.csv**: 
**processed/'site'_cloudless.csv**: 
**processed/'site'_filtered_cloudless.csv**: 


