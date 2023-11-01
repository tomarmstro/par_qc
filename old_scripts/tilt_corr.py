import numpy as np
import pandas as pd
import csv

#config
# input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\daviesA_19092017_14062021\python_output_corrA.csv'
input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\processed\python_output_corrA.csv'

output_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\processed\python_output_tilt.csv'
def tilt():
    print("Running tilt...")
    # n = 90412

    data = pd.read_csv(input_file, header=None)
    n = len(data.index)
    # openr,10,'c:\data\AIMS\rv_correction\Davies Reef\daviesB_19092017_14062021.txt',data
    # openw,11,'c:\data\AIMS\rv_correction\Davies Reef\tilt_19092017_14062021.txt'

    dn1 = data[0]
    dn = data[1]
    day = data[2]
    mo = data[3]
    yr = data[4]
    hr = data[5]
    minute = data[6]
    z = data[7]
    modpar = data[8]
    rawpar = data[9]

    dnstart = dn[0]
    yrstart = yr[0]
    dayold = day[0]
    mold = mo[0]
    yrold = yr[0]
    ii = 0
    iold = 0
    with open(output_file, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        for i in range(n):
            dn10 = dn1[i]
            dn0 = dn[i]
            mo0 = mo[i]
            yr0 = yr[i]
            day0 = day[i]

            if day0 != dayold:
                nn = (ii - 1) - (iold) + 1
                x = rawpar[iold:ii]
                y = modpar[iold:ii]
                i1 = np.argmax(x)
                i2 = np.argmax(y)
                # res_raw = x[i1]
                # res_mod = y[i2]
                del_val = i2 - i1
                #Convert everything to a dataframe, run rolling mean on the del_val column
                # mov_av = pd.Series(del_val).rolling(14).mean()

                # mov_av = del_val.rolling(14, min_periods=1)['score'].mean()
                output = (dn1[i - 1], dn[i - 1], dayold, mo[i - 1], yr[i - 1], del_val)
                writer.writerow(output)

                dayold = day0
                mold = mo0
                yrold = yr0
                iold = ii
            ii += 1
    # data2 = pd.read_csv(output_file, header=None)

    # f_output.close()


tilt()
