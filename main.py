import pandas as pd
import xarray as xr
from siphon.catalog import TDSCatalog
from datetime import datetime
from scipy.interpolate import griddata

# Function to interpolate data at a single coordinate
def interpolate_data(dataset, lon, lat):
    lons, lats = dataset.longitude.values, dataset.latitude.values
    points = list(zip(lons.ravel(), lats.ravel()))
    values = dataset.to_array().values.ravel()

    interpolated_value = griddata(points, values, (lon, lat), method='linear')
    return interpolated_value[0]

# Function to access and process data
def process_nci_data(url):
    catalog = TDSCatalog(url)
    datasets = catalog.datasets

    # Filter datasets for the year 2020
    datasets_2020 = [ds for ds in datasets if datetime.strptime(ds, "%Y%m%d%H%M%S").year == 2020]

    # Initialize an empty dataframe to store the data
    df = pd.DataFrame()

    for ds_key in datasets_2020:
        dataset_url = catalog.datasets[ds_key].access_urls['OPENDAP']

        # Open the dataset using xarray
        ds = xr.open_dataset(dataset_url)

        # Interpolate data at a specific coordinate (e.g., longitude=0, latitude=0)
        interpolated_value = interpolate_data(ds, 0, 0)

        # Append the interpolated value to the dataframe
        df = pd.concat([df, pd.DataFrame({ds_key: [interpolated_value]})], axis=1)

    return df

# URL of the NCI THREDDS catalog
nci_url = "https://dapds00.nci.org.au/thredds/catalog/rv74/satellite-products/arc/der/himawari-ahi/solar/p1h/latest/catalog.xml"

# Process the data and print the resulting dataframe
result_df = process_nci_data(nci_url)
print(result_df)