"""
Collect all urls from thredds server directory

Adapted by tarmstro 10/01/2024
"""
#!/usr/bin/env python
# import calendar
# from xml.dom import minidom
# from urllib.request import urlopen
# from urllib.request import urlretrieve
# from urllib.request import HTTPError
# import pandas as pd
from config import CONFIG
from siphon.catalog import TDSCatalog
from datetime import datetime


def get_all_files(base_url, last_himawari_data_date):
    """Get the url of all files on the thredds server from the base_url according to the year/month/day directory structure.

    Args:
        base_url (string): The base url for the intended directory within the thredds server
        last_himawari_data_date (datetime): The last date in the local himawari data csv file. Saved to prevent needlessly rescraping the thredds server.

    Returns:
        all_files (list): A list of the urls of all relevant files on the thredds server.
    """
    # Initialize an empty list to store all file URLs
    all_files = []

    # Open the base catalog
    base_catalog = TDSCatalog(base_url)

    
    # Loop over the years (sub-catalogs)
    for year in base_catalog.catalog_refs:
        # Check the thredds server starting from the last date in the saved himawari dataset
        if int(year) >= last_himawari_data_date.year:
            print(f"Checking through year: {year}.")
            year_url = base_catalog.catalog_refs[year].href
            year_catalog = TDSCatalog(year_url)
            # Loop over the months (sub-catalogs)
            for month in year_catalog.catalog_refs:
                if ((int(year) == last_himawari_data_date.year 
                    and int(month) >= last_himawari_data_date.month) 
                    or int(year) > last_himawari_data_date.year):
                    print(f"Checking through month: {month}.")
                    month_url = year_catalog.catalog_refs[month].href
                    month_catalog = TDSCatalog(month_url)
                    # Loop over the days (sub-catalogs)
                    for day in month_catalog.catalog_refs:
                        if ((int(year) == last_himawari_data_date.year 
                            and int(month) == last_himawari_data_date.month 
                            and int(day) >= last_himawari_data_date.day)
                            or (int(year) == last_himawari_data_date.year 
                            and int(month) > last_himawari_data_date.month)
                            or int(year) > last_himawari_data_date.year):
                            # print(f"Checking through day: {day}.")
                            day_url = month_catalog.catalog_refs[day].href
                            day_catalog = TDSCatalog(day_url)

                            # Loop over the hours (NetCDF files)
                            for hour in day_catalog.datasets:
                                hour_file_url = day_catalog.datasets[hour].access_urls['HTTPServer']
                                all_files.append(hour_file_url)

    file_count = len(all_files)
    print(f"Found {file_count} files.")

    return all_files


def main(last_himawari_data_date):
    """Collect all relevant thredds urls.

    Args:
        last_himawari_data_date (datetime): The last date in the local himawari data csv file. Saved to prevent needlessly rescraping the thredds server.

    Returns:
        all_files (list): A list of the urls of all relevant files on the thredds server.
        file_count (int): Count of the number of files collected. 
    """
    # Specify the base URL of the THREDDS catalog
    base_url = CONFIG['THREDDS_URL']

    # Get all file URLs
    file_urls = get_all_files(base_url, last_himawari_data_date)

    file_count = len(file_urls)

    return file_urls, file_count
