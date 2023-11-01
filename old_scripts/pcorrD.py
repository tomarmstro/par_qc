# imports
import numpy as np
import pandas as pd
import csv

# config
input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\processed\python_output_corrA.csv'
output_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\processed\python_output_corrD.csv'

def pcorrD():
    print("Running pcorrD...")

    data = pd.read_csv(input_file, header=None)
    # n = 51732
    n = len(data.index)
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
    with open(output_file, 'w', newline='') as f:
        # create the csv writer for stats
        writer = csv.writer(f)
        for i in range(n):
            pratio = 0.0001221 * dn1[i] + 0.95767
            corrpar = rawpar[i] * pratio
            # f_output.write(f"{dn1[i]} {dn[i]} {day[i]} {mo[i]} {yr[i]} {minute[i]} {rawpar[i]} {corrpar}\n")
            output = (dn1[i], dn[i], day[i], mo[i], yr[i], minute[i], rawpar[i], corrpar)
            writer.writerow(output)

pcorrD()
