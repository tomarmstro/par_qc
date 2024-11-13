# Imports
import matplotlib.pyplot as plt

# config
from config import CONFIG


def build_plots(site_name, decamin_df, daily_df, cloudless_df, deployment_start_dates):
    """
    Generates and saves various plots related to Photosynthetically Active Radiation (PAR) and tilt data for a specified site.

    Args:
        site_name (str): The name of the site for which plots are generated, used to define output file names.
        decamin_df (DataFrame): DataFrame containing decamin-level (10-minute interval) PAR data, with columns such as 'date', 'rawpar', 'modpar', 'corpar', and 'himawari_resampled'.
        daily_df (DataFrame): DataFrame with daily aggregated data, including noon PAR values for 'noon_rawpar', 'noon_corpar', 'noon_modpar', and 'noon_himawari'.
        cloudless_df (DataFrame): DataFrame with data filtered for cloudless days, containing columns like 'date', 'ratio_noon_par', and 'corrected_ratio'.
        deployment_start_dates (list of datetime): List of dates marking the start of each deployment, used to add vertical reference lines on plots.

    Plots:
        - Scatter plots for PAR ratios during cloudless days.
        - Scatter plots of daily tilt values and their rolling averages.
        - Comparison plots of PAR values (raw, model, and corrected) and differences between these values.
        - Scatter plot comparing corrected PAR and Himawari satellite PAR values, and residuals.
        - Line plot showing noon PAR values from raw, corrected, model, and Himawari data.
    """
    
    output_directory = f"{CONFIG["PROCESSED_FILE_PATH"]}\{site_name}"

    # Plotting ratios
    # ratio_scatter, ratio_scatter_ax = plt.subplots(figsize=(12, 6))
    # ratio_scatter_ax.scatter(cloudless_df['date'], cloudless_df['ratio_noon_par'], label='ratio_noon_par')
    # ratio_scatter_ax.scatter(cloudless_df['date'], cloudless_df['corrected_ratio'], label='corrected_ratio')
    # ratio_scatter_ax.legend()
    # ratio_scatter_ax.title.set_text('Pratio for cloudless days')
    # for date in deployment_start_dates:
    #     ratio_scatter_ax.axvline(date ,color='r', linestyle='--')
    # ratio_scatter_plot_file = rf'{output_directory}\{site_name}_ratio_plot.png'
    # ratio_scatter.savefig(ratio_scatter_plot_file)

    # Plotting Tilt Values
    # tilt_plot, tilt_plot_ax = plt.subplots(figsize=(12, 6))
    # tilt_plot_ax.scatter(daily_df['date'], daily_df['filtered_tilt'], label='filtered_tilt', s=5)
    # tilt_plot_ax.scatter(daily_df['date'], daily_df['tilt_rolling_avg'], label='rolling_avg', s=5)
    # tilt_plot_ax.legend()
    # tilt_plot_ax.title.set_text('Daily tilt values')
    # for date in deployment_start_dates:
    #     tilt_plot_ax.axvline(date ,color='r', linestyle='--')
    # tilt_plot_file = rf'{output_directory}\{site_name}_tilt_plot.png'
    # tilt_plot.savefig(tilt_plot_file)

    # Plots Comparing PAR Values
    # par_plot, par_plot_ax = plt.subplots(6, 1, figsize=(24, 12))
    # par_plot_ax[0].scatter(decamin_df['date'], decamin_df['rawpar'], label='raw', s=1)
    # par_plot_ax[0].legend()
    # par_plot_ax[0].title.set_text('Raw Values')
    # for date in deployment_start_dates:
    #     par_plot_ax[0].axvline(date ,color='r', linestyle='--')

    # par_plot_ax[1].scatter(decamin_df['date'], decamin_df['modpar'], label='mod', s=1)
    # par_plot_ax[1].legend()
    # par_plot_ax[1].title.set_text('Mod Values')
    # for date in deployment_start_dates:
    #     par_plot_ax[1].axvline(date ,color='r', linestyle='--')

    # par_plot_ax[2].scatter(decamin_df['date'], decamin_df['corpar'], label='cor', s=1)
    # par_plot_ax[2].legend()
    # par_plot_ax[2].title.set_text('Corr Values')
    # for date in deployment_start_dates:
    #     par_plot_ax[2].axvline(date ,color='r', linestyle='--')

    # par_plot_ax[3].scatter(decamin_df['date'], (decamin_df['rawpar'] - decamin_df['corpar']), label='raw/cor diff', s=1)
    # par_plot_ax[3].legend()
    # par_plot_ax[3].title.set_text('Difference between Raw and Corrected PAR')
    # for date in deployment_start_dates:
    #     par_plot_ax[3].axvline(date ,color='r', linestyle='--')

    # par_plot_ax[4].scatter(decamin_df['date'], (decamin_df['corpar'] - decamin_df['modpar']), label= 'cor/mod diff', s=1)
    # par_plot_ax[4].legend()
    # par_plot_ax[4].title.set_text('Difference between Corrected and Model PAR')
    # for date in deployment_start_dates:
    #     par_plot_ax[4].axvline(date ,color='r', linestyle='--')

    # par_plot_ax[5].scatter(decamin_df['date'], (decamin_df['corpar'] - decamin_df['himawari_resampled']), label= 'cor/hima diff', s=1)
    # par_plot_ax[5].legend()
    # par_plot_ax[5].title.set_text('Difference between Corrected and Himawari')
    # for date in deployment_start_dates:
    #     par_plot_ax[5].axvline(date ,color='r', linestyle='--')

    # par_plot.tight_layout(pad=5.0)
    # par_plot_file = rf'{output_directory}\{site_name}_par_plot.png'
    # par_plot.savefig(par_plot_file)


    # Plotting himawari Values
    # cor_vs_hima_plot, cor_vs_hima_plot_ax = plt.subplots(figsize=(12, 6))
    # cor_vs_hima_plot_ax.scatter(decamin_df['date'], decamin_df['corpar'], label='corpar', s=5)
    # cor_vs_hima_plot_ax.scatter(decamin_df['date'], decamin_df['interpolated_value (umol m-2 s-1)'], label='hima', s=5)
    # cor_vs_hima_plot_ax.legend()
    # cor_vs_hima_plot_ax.title.set_text('cor vs hima')
    # for date in deployment_start_dates:
    #     cor_vs_hima_plot_ax.axvline(date ,color='r', linestyle='--')
    # cor_vs_hima_plot_file = rf'{output_directory}\{site_name}_cor_hima_plot.png'
    # cor_vs_hima_plot.savefig(cor_vs_hima_plot_file)

    # Plotting himawari Values difference from corpar

    # hima_cor_filt_df = decamin_df.dropna(subset=['corpar', 'interpolated_value (umol m-2 s-1)'])
    # hima_cor_residual = hima_cor_filt_df['corpar'] - hima_cor_filt_df['interpolated_value (umol m-2 s-1)']
    # cor_vs_hima_residual_plot, cor_vs_hima_residual_plot_ax = plt.subplots(figsize=(12, 6))
    # cor_vs_hima_residual_plot_ax.scatter(hima_cor_filt_df['date'], hima_cor_residual, label='resid', s=5)
    # # hima_plot_ax2.scatter(hima_cor_filt_df['date'], hima_cor_filt_df['interpolated_value (umol m-2 s-1)'], label='hima', s=5)
    # cor_vs_hima_residual_plot_ax.legend()
    # cor_vs_hima_residual_plot_ax.title.set_text('cor vs hima resid')
    # for date in deployment_start_dates:
    #     cor_vs_hima_residual_plot_ax.axvline(date ,color='r', linestyle='--')
    # cor_vs_hima_residual_plot_file = rf'{output_directory}\{site_name}_cor_hima_residual_plot.png'
    # cor_vs_hima_residual_plot.savefig(cor_vs_hima_residual_plot_file)

    # Plotting himawari Values resampled
    # himawari_resampled_plot, himawari_resampled_plot_ax = plt.subplots(figsize=(12, 6))
    # himawari_resampled_plot_ax.scatter(decamin_df['date'], decamin_df['corpar'], label='corpar', s=5)
    # himawari_resampled_plot_ax.scatter(decamin_df['date'], decamin_df['himawari_resampled'], label='hima resamp', s=5)
    # himawari_resampled_plot_ax.legend()
    # himawari_resampled_plot_ax.title.set_text('cor vs hima resamp')
    # for date in deployment_start_dates:
    #     himawari_resampled_plot_ax.axvline(date ,color='r', linestyle='--')
    # cor_vs_hima_residual_plot_file = rf'{output_directory}\{site_name}_himawari_resampled_plot.png'
    # himawari_resampled_plot.savefig(cor_vs_hima_residual_plot_file)

    # Plotting the line plot
    himawari_filtered_df = daily_df[daily_df['noon_himawari'].notna()]

    plt.figure(figsize=(10, 6))
    plt.plot(himawari_filtered_df['date'], himawari_filtered_df['noon_rawpar'], label='noon_rawpar')
    plt.plot(himawari_filtered_df['date'], himawari_filtered_df['noon_corpar'], label='noon_corpar')
    plt.plot(himawari_filtered_df['date'], himawari_filtered_df['noon_modpar'], label='noon_modpar')
    plt.plot(himawari_filtered_df['date'], himawari_filtered_df['noon_himawari'], label='noon_himawari')

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('PAR')
    plt.title('Noon Himawari/RAW/COR/MOD PAR')
    plt.legend()

    # Saving the plot
    noon_pars_plot_file = f'{output_directory}\{site_name}_noon_pars_plot.png'
    plt.savefig(noon_pars_plot_file, dpi=300)

    # Plotly plots are significantly more interactive, though much heavier.
    # import plotly.express as px
    # # plotly_raw_vs_cor = px.data.iris() # replace with your own data source
    # decamin_df['raw_vs_cor_res'] = decamin_df['rawpar'] - decamin_df['corpar']
    # plotly_raw_vs_cor = px.scatter(
    #     decamin_df, x="date", y="raw_vs_cor_res", 
    #     # color="species"
    #     )
    # plotly_raw_vs_cor.write_html(f"{output_directory}\{site_name}_par_raw_vs_cor.html")

    print("Created plots.")