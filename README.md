
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