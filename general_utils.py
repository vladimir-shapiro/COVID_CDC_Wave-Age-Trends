import pandas as pd
import numpy as np
import csv, scipy
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import r2_score
from copy import deepcopy
from datetime import datetime
from sodapy import Socrata

def read_csv(file_path):
    rows = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(deepcopy(row))
    return rows

def get_CDC_data_Socrates(main_cont, page_cont):
    # retrieves data from  "https://data.cdc.gov/resource/vsak-wrfu.json"
    client      = Socrata(main_cont, None)
    response    = client.get(page_cont, limit=2000)
    return     response

def linear_regression(x, y, prob):
    """
    Code is adopted from from https://gist.github.com/riccardoscalco/5356167
    Return the linear regression parameters and their <prob> confidence intervals. y=b0+b1*x
    ex:
    >>> linear_regression([.1,.2,.3],[10,11,11.5],0.95)
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    xy = x * y
    xx = x * x

    # estimates
    b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean() ** 2)
    b0 = y.mean() - b1 * x.mean()
    s2 = 1. / n * sum([(y[i] - b0 - b1 * x[i]) ** 2 for i in range(n)]) # originally xrange(n)])
    print ('b0 = ', b0, end = " ")
    print ('b1 = ', b1, end = " ")
    print ('s2 = ', s2)

    # confidence intervals
    alpha = 1 - prob
    c1 = scipy.stats.chi2.ppf(alpha / 2., n - 2)
    c2 = scipy.stats.chi2.ppf(1 - alpha / 2., n - 2)
    print ('the confidence interval of s2 is: ', [n * s2 / c2, n * s2 / c1])

    c = -1 * scipy.stats.t.ppf(alpha / 2., n - 2)
    bb1 = c * (s2 / ((n - 2) * (xx.mean() - (x.mean()) ** 2))) ** .5
    print( 'the confidence interval of b1 is: ', [b1 - bb1, b1 + bb1])

    bb0 = c * ((s2 / (n - 2)) * (1 + (x.mean()) ** 2 / (xx.mean() - (x.mean()) ** 2))) ** .5
    print ('the confidence interval of b0 is: ', [b0 - bb0, b0 + bb0])
    return [b1 - bb1, b1 + bb1]

def compute_CI(sample): # compute confidence interval
    confidence_level = 0.95
    degrees_freedom = sample.size - 1
    sample_mean = np.mean(sample)
    sample_standard_error = scipy.stats.sem(sample)
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    return np.array(confidence_interval)

def get_week_range(json_dict, week_range): # restrict the data to the the primary data for the week range
    subset_data = []
    for week in json_dict:
        if week["sex"] != "All Sex" and week["age_group"] != "All ages":
            if int(week_range["start_week_num"]) <= int(week["mmwr_week"]) < int(week_range["end_week_num"]):
                subset_data.append(deepcopy(week))
    return subset_data

def parse_rename_age_group(age_group, list_groups):
    if "Under" in age_group:
        return "0_4"
    if "over" in age_group:
        return "85_OVER"

    token_space = age_group.split(" ")[0]
    token_dash = token_space.split("-")[1] # extract 14 in 5-14
    matching = [group for group in list_groups if token_dash in group]
    return matching[0]

def reformat_data(week_data, df_general_population):
    try:
        tmp1 = week_data[0]['week_ending_date'] # UTC time
        ts = datetime.strptime(tmp1, '%Y-%m-%dT%H:%M:%S.%f').strftime('%d-%m-%y')

        # initialise data of lists.
        data = { 'Age_groups': ["0_4", "5_14", "15_24",  "25_34","35_44","45_54",  "55_64","65_74","75_84",  "85_OVER"],
                 'Age_mid_range':   [2.5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                 'Males':           [0, 0, 0,  0, 0, 0, 0, 0, 0, 0],
                 'Females':         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'Both_sexes':      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'M_Total_Deaths':  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'F_Total_Deaths':  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'M_nonCOVID_Deaths': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'F_nonCOVID_Deaths': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'Total_Deaths':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'Total_nonCOVID_Deaths': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'End_week':        [ts, ts, ts, ts, ts, ts, ts, ts, ts, ts],
                 'Number_week':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 }

        # Creates pandas DataFrame.
        df = pd.DataFrame(data, index=["0_4", "5_14", "15_24", "25_34",
                                        "35_44", "45_54", "55_64", "65_74", "75_84", "85_OVER"])
        for week in week_data:
            if "All" in week["age_group"]:
                continue #skip "All ages"
            age_group = parse_rename_age_group(week["age_group"], ["0_4", "5_14", "15_24", "25_34",
                                                            "35_44", "45_54", "55_64", "65_74", "75_84", "85_OVER"])
            if week["sex"] ==  "Male":
                df_row = df.loc[df['Age_groups'] == age_group]
                df_row["Males"]             += int(week["covid_19_deaths"])
                df_row['M_Total_Deaths']    +=  int(week["total_deaths"])
                df_row['M_nonCOVID_Deaths'] = df_row['M_Total_Deaths'] - df_row["Males"]
            else:
                df_row = df.loc[df['Age_groups'] == age_group]
                df_row["Females"]           += int(week["covid_19_deaths"])
                df_row['F_Total_Deaths']    +=  int(week["total_deaths"])
                df_row['F_nonCOVID_Deaths'] = df_row['F_Total_Deaths'] - df_row["Females"]
            df.at[df['Age_groups'] == age_group] = df_row
        df["Number_week"]               = week['mmwr_week']
        df["Both_sexes"]                = df["Males"] + df["Females"]
        # eliminate commas, convert strings to ints
        df["Male_total_population"]     = df_general_population["Male_total_population"][:-1].values
        df['Male_total_population']     = df['Male_total_population'].str.replace(',', '').astype(int)
        df["Female_total_population"]   = df_general_population["Female_total_population"][:-1].values
        df['Female_total_population']   = df['Female_total_population'].str.replace(',', '').astype(int)

        df['Total_population']          = df['Male_total_population'] + df['Female_total_population']
        df['Male_mortality_relative']   = 100 * df['Males'] / df['Male_total_population']
        df['Female_mortality_relative'] = 100 * df['Females'] / df['Female_total_population']
        df['Both_sexes_relative']       = (df['Female_mortality_relative'] + df['Male_mortality_relative']) / 2
        df['Male_nonCOVID_mortality_relative']      = 100 * (df['M_Total_Deaths'] - df['Males']) / \
                                                        df['Male_total_population']
        df['Female_nonCOVID_mortality_relative']    = 100 * (df['F_Total_Deaths'] - df['Females']) / \
                                                        df['Female_total_population']
        df["Total_Deaths"]                          = df["M_Total_Deaths"] + df["F_Total_Deaths"]
        df["Total_nonCOVID_Deaths"]                 = df["M_nonCOVID_Deaths"] + df["F_nonCOVID_Deaths"]
        df['Total_COVID_Deaths_relative']           = df["Both_sexes_relative"]
        df["Total_nonCOVID_Deaths_relative"]        =  (df['Male_nonCOVID_mortality_relative'] +
                                                        df['Female_nonCOVID_mortality_relative'])/2
        return df

    except Exception as e:
        print(f"Something else went wrong in reformat_data(), error {e}")
        raise

def sumup_covid_and_total_deaths(week_data, sex):
    Total_covid = 0
    Total_all = 0
    for age in week_data:
        if age["sex"] == sex and age["age_group"] == "All Ages":
            Total_covid += int(age["covid_19_deaths"])
            Total_all   += int(age["total_deaths"])
    return Total_covid, Total_all

def calc_weekly_COVID_death_slopes_model(weeks_dicts, start_week, early_age_cutoff):
    try:
        #normalize data to the weekly total deaths
        _covid_model_dict = {
            "all_time_fraction_o65": 0,
            "all_time_slope_average": 0,
            "slopes": [],
            "models": [],
            "covid_rel_deaths": [],
            "slopes_CI_min": [], "slopes_CI_max": [],
            "slope_week": { "model1": [], "model1_r2": 0, "model2": [], "model2_r2": 0},
            "week_fraction_o65": [],
            "week_fraction_o65_CI_min": [], "week_fraction_o65_CI_max": [],
            "week_total_deaths": [],
        }

        _non_covid_model_dict = {
            "slopes": [],
            "slope_week": {"model1": [], "model1_r2": 0, "model2": [], "model2_r2": 0},
            "week_fraction_o65": [],
        }

        # obtain a list of relevant ages
        _ages = []
        week = weeks_dicts[0] # any week would do
        for idx, age in enumerate(week.Age_groups):
            if week.Age_mid_range[idx] < early_age_cutoff or sum(week.Both_sexes) == 0:
                continue  # ignore early ages
            _ages.append(deepcopy(week.Age_mid_range[idx]))

        _end_week                            = []
        all_time_covid_fraction_o65_sum     = 0
        all_time_covid_fraction_o65_count   = 0
        all_time_slope_sum                  = 0
        all_time_slope_count                = 0
        for week_idx, week in enumerate(weeks_dicts):
            this_week_abs_num = week_idx + start_week
            print(f"\nWeek={week.End_week[0]}, #{this_week_abs_num}", end = " ")
            _end_week.append(deepcopy(week["End_week"][0]))
            covid_rel_deaths                = []
            non_covid_rel_deaths            = []
            sum_covid_o65                   = 0
            sum_non_covid_o65               = 0
            for idx, age in enumerate(week.Age_groups):
                if week.Age_mid_range[idx] < early_age_cutoff or sum(week.Both_sexes) == 0:
                    continue  #ignore early ages
                covid_rel_death_count          = round(np.log10(week["Both_sexes_relative"][age]), 5)
                non_covid_rel_death_count      = np.log10(week["Total_nonCOVID_Deaths_relative"][age])
                covid_rel_deaths.append(deepcopy(covid_rel_death_count))
                non_covid_rel_deaths.append(deepcopy(non_covid_rel_death_count))
                if age == '65_74' or age == '75_84' or age == '85_OVER':
                    sum_covid_o65       += week["Both_sexes"][age]
                    sum_non_covid_o65   += week["Total_nonCOVID_Deaths"][age]

            fraction_covid_o65                  = sum_covid_o65 * 100/sum(week.Both_sexes)
            fraction_non_covid_o65              = sum_non_covid_o65 * 100/sum(week.Total_nonCOVID_Deaths)
            all_time_covid_fraction_o65_sum     += fraction_covid_o65
            all_time_covid_fraction_o65_count   += 1
            _covid_model_dict["week_total_deaths"].append(deepcopy(sum(week.Total_Deaths) -
                                                            sum(week.Total_nonCOVID_Deaths)))
            _covid_model_dict["week_fraction_o65"].append(deepcopy(fraction_covid_o65))
            _non_covid_model_dict["week_fraction_o65"].append(deepcopy(fraction_non_covid_o65))

            # CI
            # Function for computing confidence intervals - Confidence interval for population proportion
            CI_min, CI_max = proportion_confint(count=sum_covid_o65,  # Number of "successes"
                                               nobs=sum(week.Both_sexes),  # Number of trials
                                               alpha=0.05)
            _covid_model_dict["week_fraction_o65_CI_max"].append(deepcopy(CI_max * 100))
            _covid_model_dict["week_fraction_o65_CI_min"].append(deepcopy(CI_min * 100))

            [b1_CI_0, b1_CI_1] = linear_regression(np.array(_ages), np.array(covid_rel_deaths), 0.95)
            _covid_model_dict["slopes_CI_max"].append(deepcopy(b1_CI_1))
            _covid_model_dict["slopes_CI_min"].append(deepcopy(b1_CI_0))
            covid_model1      = np.polyfit(np.array(_ages), np.array(covid_rel_deaths), 1, full=False) # m - slope, b - intercept
            covid_model1_r2   = r2_score(covid_rel_deaths, np.polyval(covid_model1, _ages))
            covid_model2      = np.polyfit(np.array(_ages), np.array(covid_rel_deaths), 2)
            covid_model2_r2   = r2_score(covid_rel_deaths, np.polyval(covid_model2, _ages))

            _covid_model_dict["covid_rel_deaths"].append(deepcopy(covid_rel_deaths))
            _covid_model_dict["slope_week"]["covid_model1"]      = covid_model1
            _covid_model_dict["slope_week"]["covid_model1_r2"]   = covid_model1_r2
            _covid_model_dict["slope_week"]["covid_model2"]      = covid_model2
            _covid_model_dict["slope_week"]["covid_model2_r2"]   = covid_model2_r2
            _covid_model_dict["slopes"].append(deepcopy(covid_model1[0]))
            _covid_model_dict["models"].append(deepcopy(covid_model1))
            all_time_slope_sum                          += covid_model1[0]
            all_time_slope_count                        += 1

            non_covid_model1 = np.polyfit(np.array(_ages), np.array(non_covid_rel_deaths), 1, full=False)
            _non_covid_model_dict["slopes"].append(deepcopy(non_covid_model1[0]))
            print(f" Log-poly coef cov-slope={round(covid_model1[0],3)} non-cov-slope={round(non_covid_model1[0],3)} ", end =" " ),

        _covid_model_dict["all_time_fraction_o65"]     = all_time_covid_fraction_o65_sum / all_time_covid_fraction_o65_count
        _covid_model_dict["all_time_slope_average"]    = all_time_slope_sum / all_time_slope_count
        return _covid_model_dict, _non_covid_model_dict, _end_week, _ages
    except Exception as e:
      print(f"Something else went wrong in calc_weekly_COVID_death_slopes_model(), error {e}")

def calc_weekly_COVID_death_average(weeks_dicts):
    #normalize data to the weekly total deaths
    _covid_avg_dict = {
        "all_time_age_average_weighted": 0,
        "all_time_age_average_non_weighted": 0,
        "age_averages": [],
        "age_averages_CI_min": [], "age_averages_CI_max": [],
        "total_deaths": [],
    }

    _non_avg_covid_dict = {
        "age_averages": [],
        "total_deaths":  []
    }
    _week_nums                   = []
    all_time_age_average_sum    = 0
    all_time_age_average_count  = 0
    all_time_total_composites   = 0
    all_time_total_weights      = 0
    for week_idx, week in enumerate(weeks_dicts):
        print(f"\nWeek={week.End_week[0]} #{week_idx}", end = " ")
        total_non_covid_composites  = 0
        total_non_covid_weights     = 0
        total_composites            = 0
        total_weights               = 0
        covid_raw                   = [] # raw sample
        for idx, age in enumerate(week.Age_groups):
            if idx < 3 or sum(week.Both_sexes) == 0:
                continue  #ignore early ages
            total_non_covid_composites  += week["Total_nonCOVID_Deaths"][age] * week['Age_mid_range'][age]
            total_non_covid_weights     += week["Total_nonCOVID_Deaths"][age]

            both_fr = round(week["Both_sexes"][age] , 5)
            total_composites    += both_fr * week['Age_mid_range'][age]
            total_weights       += both_fr
            for i in range(0, week["Both_sexes"][age]):
                covid_raw.append(deepcopy(week['Age_mid_range'][age]))
        all_time_total_composites   += total_composites
        all_time_total_weights      += total_weights
        average_age_CI              = compute_CI(np.array(covid_raw))
        _covid_avg_dict["age_averages_CI_min"].append(deepcopy(min(average_age_CI)))
        _covid_avg_dict["age_averages_CI_max"].append(deepcopy(max(average_age_CI)))
        non_covid_average_age       = total_non_covid_composites / total_non_covid_weights
        average_age                 = total_composites / total_weights
        all_time_age_average_sum    += average_age
        all_time_age_average_count  += 1

        _non_avg_covid_dict["total_deaths"].append(deepcopy(sum(week.Total_nonCOVID_Deaths)))
        _covid_avg_dict["total_deaths"].append(deepcopy(sum(week.Both_sexes)))

        _covid_avg_dict["age_averages"].append(deepcopy(average_age))
        _non_avg_covid_dict["age_averages"].append(deepcopy(non_covid_average_age))
        _week_nums.append(int(week['Number_week'].values[0]))

    _covid_avg_dict["all_time_age_average_weighted"]     =  all_time_total_composites / all_time_total_weights
    _covid_avg_dict["all_time_age_average_non_weighted"] =  all_time_age_average_sum / all_time_age_average_count
    return _week_nums, _covid_avg_dict, _non_avg_covid_dict
