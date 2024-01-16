#!/usr/bin/env python
# Script to download all .nc files from a THREDDS catalog directory
# Written by Sage 4/5/2016, revised 5/31/2018

import calendar
from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve

# START_YEAR = 2019
# START_YEAR_MONTH = 4
START_YEAR = 2021
START_YEAR_MONTH = 10

# END_YEAR = 2023
END_YEAR = 2021
END_YEAR_MONTH = 9

years = list(range(START_YEAR, (END_YEAR + 1)))


def get_thredds_file_urls():
    file_urls = []
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
            for day in days:
                date_url = f"{year:04}" + '/' + f"{month:02}" + '/' + f"{day:02}" + '/'

                file_urls.append(get_filename(date_url))
    return file_urls
    # df = pd.DataFrame(file_urls)
    # print(file_urls)
    # saving the dataframe
    # df.to_csv('test.csv')

# https://dapds00.nci.org.au/thredds/dodsC/rv74/satellite-products/arc/der/himawari-ahi/solar/p1d/latest/2020/01/01/IDE02326.202001010000.nc


def get_elements(url, tag_name, attribute_name):
    """Get elements from an XML file"""
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


def get_filename(date_url):
    server_url = 'https://dapds00.nci.org.au/thredds/'
    request_url = 'rv74/satellite-products/arc/der/himawari-ahi/solar/p1d/latest/'
    url = server_url + request_url + date_url + 'catalog.xml'
    #     print(url)
    catalog = get_elements(url, 'dataset', 'urlPath')
    files = []
    for citem in catalog:
        if (citem[-3:] == '.nc'):
            files.append(citem)
    count = 0
    for f in files:
        count += 1
        file_url = server_url + 'fileServer/' + f
        file_prefix = file_url.split('/')[-1][:-3]
        # file_name = file_prefix + '_' + str(count) + '.nc'
        file_name = file_prefix + '.nc'
        file_url = server_url + 'dodsC/' + request_url + date_url + file_name

        # print(file_name)
        # print('Downloading file %d of %d' % (count,len(files)))
        # a = urlretrieve(file_url,file_name)
        # print(a)
        return file_url

# print('Downloading file %d of %d' % (count,len(files)))
# a = urlretrieve('https://dapds00.nci.org.au/thredds/' + 'fileServer/' + 'IDE02327.201903311830.nc')
# print(a)

# Run main function when in comand line mode
# if __name__ == '__main__':
#     main()