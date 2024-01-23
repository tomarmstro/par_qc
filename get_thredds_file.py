#!/usr/bin/env python
# Script to download all .nc files from a THREDDS catalog directory
# Written by Sage 4/5/2016, revised 5/31/2018
# adapted by tarmstro 10/01/2024

import calendar
from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve

# Config
# Set data interval "daily" or "hourly"
# DATA_INTERVAL = 'daily'
DATA_INTERVAL = 'hourly'

# START_YEAR = 2019
# START_YEAR_MONTH = 4
START_YEAR = 2022
START_YEAR_MONTH = 12

# END_YEAR = 2023
END_YEAR = 2022
# END_YEAR_MONTH = 9
END_YEAR_MONTH = 12

years = list(range(START_YEAR, (END_YEAR + 1)))


def get_thredds_file_urls(data_interval='daily'):
    file_urls = []
    # Iterate through all the days of the months of the years given by config values and generate a list of files
    for year in years:
        if year == START_YEAR:
            months = list(range(START_YEAR_MONTH, 13))
        elif year == END_YEAR:
            months = list(range(1, END_YEAR_MONTH))
        else:
            months = list(range(1, 13))
        for month in months:
            print(f"Getting files for {calendar.month_name[month]}, {year}.")
            days = list(range(1, (calendar.monthrange(year, month)[1] + 1)))
            # days = list(range(1, 3))
            for day in days:
                date_url = f"{year:04}" + '/' + f"{month:02}" + '/' + f"{day:02}" + '/'
                file_urls.append(get_filename(date_url, data_interval))
    if data_interval == 'hourly':
        file_count = 0
        for hourly_file in file_urls:
            file_count += len(hourly_file)
    elif data_interval == 'daily':
        file_count = len(file_urls)
    print(f"Found {file_count} {data_interval} files of  data between {calendar.month_name[START_YEAR_MONTH]}, {START_YEAR} and {calendar.month_name[END_YEAR_MONTH]}, {END_YEAR}.")
    return file_urls, file_count


def get_elements(url, tag_name, attribute_name):
    # Get elements for our catalog from XML file
    # usock = urllib2.urlopen(url)
    usock = urlopen(url)
    xmldoc = minidom.parse(usock)
    usock.close()
    tags = xmldoc.getElementsByTagName(tag_name)
    attributes = []
    for tag in tags:
        attribute = tag.getAttribute(attribute_name)
        attributes.append(attribute)
    return attributes

# Get filenames from the thredds server catalog
def get_filename(date_url, data_interval):
    # Check config for sample interval
    if data_interval == 'daily':
        interval_str = 'p1d/'
    elif data_interval == 'hourly':
        interval_str = 'p1h/'
    # Thredds server specific urls
    server_url = 'https://dapds00.nci.org.au/thredds/'
    request_url = 'rv74/satellite-products/arc/der/himawari-ahi/solar/' + interval_str + 'latest/'
    url = server_url + request_url + date_url + 'catalog.xml'
    # Run get_elements func to find catalog metadata
    catalog = get_elements(url, 'dataset', 'urlPath')
    files = []
    # Iterate through the generated catalog to find all netcdf files
    for file in catalog:
        if (file[-3:] == '.nc'):
            # print(citem)
            if data_interval == 'daily':
                files.append(server_url + 'dodsC/' + file)
            elif data_interval == 'hourly':
                files.append(server_url + 'dodsC/' + file)
    return files

# a = urlretrieve('https://dapds00.nci.org.au/thredds/' + 'fileServer/' + 'IDE02327.201903311830.nc')

# get_thredds_file_urls()
