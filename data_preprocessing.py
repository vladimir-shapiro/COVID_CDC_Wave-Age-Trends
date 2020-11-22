from copy import deepcopy
from general_utils import get_week_range,  reformat_data, sumup_covid_and_total_deaths

def data_preprocessing(json_dict, df, _start_week, _end_week):
    weeks_dicts    = []
    total_covid_deaths  = 0
    total_deaths        = 0
    for idx in range(_start_week, _end_week) :# week in enumerate(whole_json_dict):
        weeks_roi = {"start_week_num": str(idx), "end_week_num": str(idx + 1)}
        week_data = get_week_range(json_dict, weeks_roi)
        if len( week_data):
            df_week = reformat_data(week_data, df)

            weeks_dicts.append(deepcopy(df_week))
            total_male_covid_deaths, total_male_all_deaths      = sumup_covid_and_total_deaths(week_data, "Male")
            total_female_covid_deaths, total_female_all_deaths  = sumup_covid_and_total_deaths(week_data, "Female")
            total_covid_deaths  += total_male_covid_deaths + total_female_covid_deaths
            total_deaths        += total_male_all_deaths + total_female_all_deaths
            # df_week.to_csv(os.path.join(DATA_FOLDER, f"US_week_{idx}.csv")) # save weekly data if needed
            print(f"\tWeek #{idx}, ending at {week_data[0]['week_ending_date']}: "
                    f"total - {total_male_all_deaths + total_female_all_deaths} "
                    f"COVID - {total_male_covid_deaths + total_female_covid_deaths}")

    print(f"Total deaths between weeks {_start_week} and {_end_week} are {total_deaths}. COVID-19 deaths {total_covid_deaths}")

    return weeks_dicts