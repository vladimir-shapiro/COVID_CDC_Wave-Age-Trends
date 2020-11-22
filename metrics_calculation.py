import numpy as np
from copy import deepcopy
from sklearn.metrics import r2_score
from statsmodels.stats.proportion import proportion_confint
from general_utils import read_csv, get_CDC_data_Socrates, linear_regression, compute_CI, \
    get_week_range,  reformat_data, sumup_covid_and_total_deaths


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