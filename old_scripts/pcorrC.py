import numpy as np
from scipy.stats import zscore
import pandas as pd
import csv
import matplotlib.pyplot as plt

#config
input_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\processed\python_output_corrB_cloudless.csv'
output_file = r'C:\Users\tarmstro\Python\par_qc\manuel_files\processed\python_output_corrC.csv'

def pcorrC():

    print("Running pcorrC...")
    # n = 142

    data = pd.read_csv(input_file, header=None)
    n = len(data.index)
    x = data[0]
    dn = data[1]
    day = data[2]
    mo = data[3]
    yr = data[4]
    y = data[5]

    # Sort data by x values
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Fit least squares regression: y vs x
    v = np.polyfit(x, y, 1)

    # Print coefficients of the fitted line (v[0] is the intercept, v[1] is the slope)
    # print(v[0])
    # print(v[1])

    # Removing data greater than 2 stdev from the fit
    while True:
        residuals = y - (v[0] + v[1] * x)
        z_scores = zscore(residuals)
        filtered_indices = np.abs(z_scores) <= 2.0

        if np.sum(filtered_indices) == len(x):
            break

        x = x[filtered_indices]

        y = y[filtered_indices]
        v = np.polyfit(x, y, 1)

    # Print final coefficients of the fitted line after removing outliers
    # print("Final coefficients after removing outliers:")
    # print(v[0])
    # print(v[1])
    corrected_ratios = []
    with open(output_file, 'w', newline='') as f:
        # create the csv writer
        output_writer = csv.writer(f)
        # for row in data.index:
        for row in range(n):
            #Missing rows breaks the iteration
            try:
                x2 = x[row]
            except KeyError:
                continue
            dn2 = dn[row]
            day2 = day[row]
            mo2 = mo[row]
            yr2 = yr[row]
            y2 = y[row]
            corrected_ratio = 0.0001221 * x2 + 0.95767 # (should we be using v values here?)
            # print("corrected ratio is 0.0001221 * ", x2, " + 0.95767") # should this be x or y?
            corrected_ratios.append(corrected_ratio)
            output = (x2, dn2, day2, mo2, yr2, y2, corrected_ratio)
            output_writer.writerow(output)

    plt.scatter(x,y)
    # plt.scatter(x,corrected_ratios, "b")
    # plt.plot(x, v[0] * x + v[1], "r") # (m*x + b)
    plt.plot(x, corrected_ratios, "r")  # (m*x + b)
    plt.show()
pcorrC()