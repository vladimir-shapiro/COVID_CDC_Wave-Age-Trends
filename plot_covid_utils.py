import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import addcopyfighandler # helps copying the figure to the clipboard, as ctrl-C and ctrl-V
from matplotlib.cm import get_cmap

###############################################################
# Auxiliary functions
###############################################################
def moving_average(x, n=3):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if n%2 == 0:
            a, b = i - (n-1)//2, i + (n-1)//2 + 2
        else:
            a, b = i - (n-1)//2, i + (n-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out

def set_rcParams():
    plt.rcParams['axes.labelsize']      = 8
    plt.rcParams['xtick.labelsize']     = 7
    plt.rcParams['ytick.labelsize']     = 7
    plt.rcParams['lines.linewidth']     = 0.8
    plt.rcParams['figure.titlesize']    = 11
    plt.rcParams['figure.figsize']      = [9, 6]
    plt.rcParams['figure.dpi']          = 200 
    plt.rcParams['figure.facecolor']    = 'white'
    plt.rc('legend', fontsize=6)


###############################################################
# Weekly trends and death counts
###############################################################
def plot_mortality_65plus(covid_dict, covid_dict1, end_week):
    set_rcParams()
    fig, ax1 = plt.subplots()
    ax1.title.set_text("COVID-19 Mortality 65+ Fraction (%)")
    ax1.title.set_size(10)
    ax1.set_xlabel('Last week day')
    ax1.set_ylabel('65+ Fraction (%)', color='red')  # we already handled the x-label with ax1
    ax1.set_ylabel('Age', color='red')
    fraction65_moving_av =  moving_average(np.array(covid_dict1["week_fraction_o65"]), n=5)
    ax1.plot(np.array(end_week), fraction65_moving_av, color='red', linestyle='dashed',
              label='Fraction 65+ deaths (%) (moving average)')
    plt.axhline(y=covid_dict1["all_time_fraction_o65"], color='red', linestyle='dotted', label='All Time Average')
    ax1.plot(np.array(end_week), covid_dict1["week_fraction_o65"], color='red',  label='Fraction 65+ deaths (%)')
    ax1.fill_between(np.array(end_week), np.array(covid_dict1["week_fraction_o65_CI_min"]),
                     np.array(covid_dict1["week_fraction_o65_CI_max"]), color='red', alpha=0.1)

    ax1.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='lower left')
    plt.xticks(rotation=90, fontsize=7)
    plt.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('COVID-19 Death Count', color='black')  # we already handled the x-label with ax1
    ax2.plot(np.array(end_week), np.array(covid_dict["total_deaths"]), color='black', linestyle='solid',
             label='COVID-19 Death Count')
    ax2.fill_between(np.array(end_week), np.array(covid_dict["total_deaths"]), 0, color='black', alpha=0.1)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='lower right')
    fig.tight_layout(pad=2)  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_mortality_age_structure(covid_dict, end_week):
    set_rcParams()
    fig, ax1 = plt.subplots()
    ax1.title.set_text("COVID-19 Mortality Average Age")
    ax1.title.set_size(10)
    ax1.set_xlabel('Last week day')
    ax1.set_ylabel('Age', color='red')
    age_averages_moving_av =  moving_average(np.array(covid_dict["age_averages"]), n=5)
    plt.plot(np.array(end_week), age_averages_moving_av, color="red", linestyle='dashed',
              label="Average Age (Moving Average) ")
    plt.axhline(y=covid_dict["all_time_age_average_weighted"], color='red', linestyle='dotted', label='All Time Average')
    ax1.fill_between(np.array(end_week), np.array(covid_dict["age_averages_CI_min"]),
                     np.array(covid_dict["age_averages_CI_max"]), color='red', alpha=0.1)
    ax1.plot(np.array(end_week), np.array(covid_dict["age_averages"]), color='red',  label='Average age')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='lower left')
    plt.xticks(rotation=90, fontsize=7)
    plt.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('COVID-19 Death Count', color='black')  # we already handled the x-label with ax1
    ax2.plot(np.array(end_week), np.array(covid_dict["total_deaths"]), color='black', linestyle='solid',
             label='COVID-19 Death Count')
    ax2.fill_between(np.array(end_week), np.array(covid_dict["total_deaths"]), 0, color='black', alpha=0.1)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='lower right')
    fig.tight_layout(pad=2)  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_layered_65plus(covid_dict, end_week):
    set_rcParams()
    # layered
    y = np.array([np.array(covid_dict["week_total_deaths"]) * (1. - np.array(covid_dict["week_fraction_o65"]) / 100),
                  np.array(covid_dict["week_total_deaths"]) * np.array(covid_dict["week_fraction_o65"]) / 100],
                 np.int32)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(np.array(end_week), y, labels=[ 'Under 65', 'Over 65'], alpha=0.75, colors=['black', 'red'])
    ax.set_title('Age Structure of COVID-19 Deaths')
    ax.title.set_size(10)
    ax.legend(loc='upper left')
    ax.set_ylabel('COVID-19 Death Count')
    ax.set_xlabel('Last week day')
    fig.tight_layout(pad=4)
    plt.xticks(rotation=90, fontsize=7)
    plt.grid()
    plt.show()

def plot_slope(end_week,covid_dict, non_covid_dict ):
    set_rcParams()
    slope_moving_av =  moving_average(np.array(covid_dict["slopes"]), n=5)
    plt.plot(np.array(end_week), np.array(covid_dict["slopes"]), color="red", label="COVID-19 ")
    plt.plot(np.array(end_week), slope_moving_av, color="red", linestyle="dashed", label="COVID-19 Moving Average ")
    plt.plot(np.array(end_week), np.array(non_covid_dict["slopes"]), color="black", label="non_COVID ")
    plt.fill_between(np.array(end_week), np.array(covid_dict["slopes_CI_min"]),
                     np.array(covid_dict["slopes_CI_max"]), color='red', alpha=0.1)
    plt.xticks(rotation=90, fontsize=7)
    plt.xlabel('Last week day')
    plt.grid()
    plt.legend()
    plt.tight_layout(pad=2)
    plt.ylabel('Log regression slope (b1)', color='red')
    plt.title("Age Gradient Slope", fontsize=10)
    plt.show()


###############################################################
# COVID vs Non-COVID plots
###############################################################
def plot_slope_vs_deaths(end_week, _covid_avg_dict, _covid_model_dict):
    set_rcParams()

    # same plot 2 series - non-covid
    fig, ax1 = plt.subplots()
    ax1.title.set_text("COVID-19 Age Gradient Slope")
    ax1.title.set_size(10)
    ax1.set_xlabel('Last week day')
    ax1.set_ylabel('Log regression slope (b1)', color='red')
    ax1.plot(np.array(end_week), np.array(_covid_model_dict["slopes"]), color='red',
             label='Log Fit Slope')
    slope_moving_av =  moving_average(np.array(_covid_model_dict["slopes"]), n=5)
    plt.plot(np.array(end_week), slope_moving_av, color="red", linestyle='dashed',
             label="Log Fit Slope (Moving Average) ")
    plt.axhline(y=_covid_model_dict["all_time_slope_average"], color='red', linestyle='dotted', label='All Time Average')
    ax1.fill_between(np.array(end_week), np.array(_covid_model_dict["slopes_CI_min"]),
                     np.array(_covid_model_dict["slopes_CI_max"]), color='red', alpha=0.1)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='lower left')
    plt.xticks(rotation=90, fontsize=7)
    plt.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('COVID-19 Death Count', color='black')  # we already handled the x-label with ax1
    ax2.plot(np.array(end_week), np.array(_covid_avg_dict["total_deaths"]), color='black', linestyle='solid',
             label='COVID-19 Death Count')
    ax2.fill_between(np.array(end_week), np.array(_covid_avg_dict["total_deaths"]), 0, color='black', alpha=0.1)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='lower right')

    fig.tight_layout(pad=2)  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_deaths_regression_slopes(list_weeks_dicts, _high_week, _low_week, _covid_model_dict, ages):
    # Raw Mortality Data
    hw              = list_weeks_dicts[_high_week]["End_week"][0]
    label_high      = f"Outbreak High ({hw})"
    lw              = list_weeks_dicts[_low_week]["End_week"][0]
    label_low       = f"Outbreak Low ({lw})"
    plt.plot(np.array(list_weeks_dicts[_high_week].Age_mid_range),
             list_weeks_dicts[_high_week].Total_COVID_Deaths_relative,
             color='red', alpha=0.75, label=label_high)
    plt.plot(np.array(list_weeks_dicts[_low_week].Age_mid_range),
             list_weeks_dicts[_low_week].Total_COVID_Deaths_relative,
             color='green', alpha=0.75, label=label_low)
    plt.xlabel('Age')
    plt.grid()
    plt.legend()
    plt.ylabel('Fatality Count per 100,000 (Standardized)')
    plt.title("Raw Age Gradient ", fontsize=10)
    plt.show()

    # Simple Regression Fit
    high_covid_model1       = _covid_model_dict["models"][_high_week]
    high_covid_rel_deaths   = _covid_model_dict["covid_rel_deaths"][_high_week]
    low_covid_model1        = _covid_model_dict["models"][_low_week]
    low_covid_rel_deaths    = _covid_model_dict["covid_rel_deaths"][_low_week]
    label_high              = f"Week of High Fit Slope ({hw})"
    label_low               = f"Week of Low Fit Slope ({lw})"
    label_covid             = f"Week of High Fit Slope (Fit): y={round(high_covid_model1[1], 2)} + " \
                              f"{round(high_covid_model1[0], 3)}x"
    label_gen = f"Week of Low Fit Slope (Fit): y={round(low_covid_model1[1], 2)} + {round(low_covid_model1[0], 3)}x"
    plt.plot(np.array(ages), np.polyval(high_covid_model1, ages), color='red', linestyle='dashed', alpha=0.25,
             label=label_covid)
    plt.plot(np.array(ages), np.polyval(low_covid_model1, ages), color='green', linestyle='dashed', alpha=0.25,
             label=label_gen)
    plt.plot(np.array(ages), high_covid_rel_deaths, color='red', alpha=0.75, label=label_high)
    plt.plot(np.array(ages), low_covid_rel_deaths, color='green', alpha=0.75, label=label_low)
    plt.xlabel('Age')
    plt.grid()
    plt.legend()
    plt.ylabel('Log10(Fatality Count)')
    plt.title("Age Gradient Regression in Log Domain", fontsize=10)
    plt.show()

###############################################################
# COVID vs Non-COVID plots
###############################################################
def plot_average_vs_noncovid(covid_avg_dict, non_avg_covid_dict, end_week):
    set_rcParams()

    plt.plot(np.array(end_week), np.array(covid_avg_dict["age_averages"]),
             color='red', label="COVID")
    plt.fill_between(np.array(end_week), np.array(covid_avg_dict["age_averages_CI_min"]),
                     np.array(covid_avg_dict["age_averages_CI_max"]), color='red', alpha=0.1)
    plt.plot(np.array(end_week), np.array(non_avg_covid_dict["age_averages"]), color="blue",
             label="Non-COVID Deaths ")

    plt.xticks(rotation=90, fontsize=7)
    plt.xlabel('Last week day')
    plt.grid()
    plt.legend()
    plt.tight_layout(pad=2)
    plt.ylabel('Age at Death (years)', color='black')
    plt.title("Average Age (COVID vs non-COVID Deaths)", fontsize=10)
    plt.show()

def plot_65plus_vs_noncovid(_covid_avg_dict, _non_avg_covid_dict, _end_week):
    set_rcParams()
    plt.plot(np.array(_end_week), np.array(_covid_avg_dict["week_fraction_o65"]), color="red", label="COVID Deaths")
    plt.plot(np.array(_end_week), np.array(_non_avg_covid_dict["week_fraction_o65"]), color="blue",
             label="Non-COVID Deaths")
    plt.fill_between(np.array(_end_week), np.array(_covid_avg_dict["week_fraction_o65_CI_min"]),
                     np.array(_covid_avg_dict["week_fraction_o65_CI_max"]), color='red', alpha=0.1)
    plt.xticks(rotation=90, fontsize=7)

    ax = plt.gca()
    major_ticks = np.arange(70, 88, 2)
    minor_ticks = np.arange(70, 88, 1)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')

    plt.xlabel('Last week day')
    plt.legend()
    plt.tight_layout(pad=2)
    plt.ylabel('Percent (%)', color='black')
    plt.title("Fraction of 65+ Ages (COVID vs non-COVID Deaths)", fontsize=10)
    plt.show()

def plot_slope_vs_noncovid(_covid_model_dict, _non_covid_model_dict, _end_week):
    set_rcParams()

    ax = plt.gca()
    ax.title.set_text("Age Gradient Slope (COVID vs non-COVID Deaths)")
    ax.title.set_size(10)
    ax.set_xlabel('Last week day')
    ax.set_ylabel('Log regression slope (b1)', color='black')
    ax.plot(np.array(_end_week), np.array(_covid_model_dict["slopes"]), color='red',
             label='COVID Deaths')
    ax.plot(np.array(_end_week), np.array(_non_covid_model_dict["slopes"]), color='blue',
             label='Non-COVID Deaths')
    ax.fill_between(np.array(_end_week), np.array(_covid_model_dict["slopes_CI_min"]),
                     np.array(_covid_model_dict["slopes_CI_max"]), color='red', alpha=0.1)
    plt.xticks(rotation=90, fontsize=7)

    ax = plt.gca()
    major_ticks = np.arange(0.03, 0.05, 0.005)
    minor_ticks = np.arange(0.03, 0.05, 0.001)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')

    plt.legend()
    plt.tight_layout(pad=2)
    plt.show()

###############################################################
# Differential of COVID over Non-COVID plots
###############################################################
def    plot_average_over_noncovid(_covid_avg_dict, _non_avg_covid_dict, _end_week):
    set_rcParams()

    avg_differential = np.array(_covid_avg_dict["age_averages"]) - np.array(_non_avg_covid_dict["age_averages"])
    plt.plot(np.array(_end_week), avg_differential, color='red', alpha=0.3, label="COVID over non-COVID differential")
    age_averages_moving_av =  moving_average(avg_differential, n=5)
    plt.plot(np.array(_end_week), age_averages_moving_av, color="red", linestyle='dashed', linewidth=1.5,
              label="Average Differential (Moving Average) ")

    plt.xticks(rotation=90, fontsize=7)
    plt.xlabel('Last week day')
    plt.grid()
    plt.tight_layout(pad=2)
    plt.ylabel('Age at Death (years)', color='black')
    plt.title("Average Age Differential (COVID over non-COVID Deaths)", fontsize=10)
    plt.show()

def    plot_65plus_over_noncovid(_covid_model_dict, _non_model_covid_dict, _end_week):
    set_rcParams()

    avg_differential = np.array(_covid_model_dict["week_fraction_o65"]) - np.array(_non_model_covid_dict["week_fraction_o65"])
    plt.plot(np.array(_end_week), avg_differential, color='red', alpha=0.3, label="COVID over non-COVID differential")
    age_averages_moving_av =  moving_average(avg_differential, n=5)
    plt.plot(np.array(_end_week), age_averages_moving_av, color="red", linestyle='dashed', linewidth=1.5,
              label="Fraction Differential (Moving Average) ")

    plt.xticks(rotation=90, fontsize=7)
    plt.xlabel('Last week day')
    plt.grid()
    plt.tight_layout(pad=2)
    plt.ylabel('Percent (%)', color='black')
    plt.title("65+ Fraction Differential (COVID over non-COVID Deaths)", fontsize=10)
    plt.tight_layout(pad=2)
    plt.show()

def plot_slope_over_noncovid(_covid_model_dict, _non_model_covid_dict, _end_week):
    set_rcParams()

    avg_differential = np.array(_covid_model_dict["slopes"]) - np.array(
        _non_model_covid_dict["slopes"])
    plt.plot(np.array(_end_week), avg_differential, color='red', alpha=0.3, label="COVID over non-COVID differential")
    age_averages_moving_av = moving_average(avg_differential, n=5)
    plt.plot(np.array(_end_week), age_averages_moving_av, color="red", linestyle='dashed', linewidth=1.5,
             label="Slope Differential (Moving Average) ")

    plt.xticks(rotation=90, fontsize=7)
    plt.xlabel('Last week day')
    plt.grid()
    plt.ylabel('Log regression slope (b1)', color='black')
    plt.title("Age Gradient Slope Differential (COVID over non-COVID Deaths)", fontsize=10)
    plt.tight_layout(pad=2)
    plt.show()
