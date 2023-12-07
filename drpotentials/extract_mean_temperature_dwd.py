import os
from typing import List

import numpy as np
import pandas as pd

from drpotentials.tools import make_directory_if_missing

FILE_PATH_IN = "D:/tubCloud2/Promotion/Literatur/Wetterdaten/DWD_Wetterdaten"
FILE_PATH_OUT = "prepared/"
YEAR = 2017
COLS = ["TT_TU"]
ERROR_VALUE = -999


def walk_data_folder_and_extract_mean(
    file_path_in: str,
    file_path_out: str,
    year: int,
    cols: List[str],
    error_value: float,
):
    """Go through data folder and extract mean hourly and daily temperature"""
    all_temperatures = pd.DataFrame()
    average_temperatures = pd.DataFrame()
    for file in os.listdir(file_path_in):
        if file.endswith(".txt"):
            stations_id, filtered = read_and_filter_weather_datum(
                os.path.join(file_path_in, file), year, cols, error_value
            )
            all_temperatures[stations_id] = filtered
    make_directory_if_missing(
        f"{file_path_in.rsplit('/', 1)[0]}/{file_path_out}"
    )
    average_temperatures["hourly_mean"] = all_temperatures.mean(axis=1)
    daily_means = pd.DataFrame()
    daily_means["same_day"] = (
        average_temperatures["hourly_mean"].resample("D").mean()
    )
    average_temperatures["daily_mean"] = daily_means["same_day"]
    daily_means["equivalent_rolling_weighted_mean"] = rolling_weighted_average(
        daily_means["same_day"]
    )
    average_temperatures["equivalent_rolling_weighted_mean"] = daily_means["equivalent_rolling_weighted_mean"]
    average_temperatures = average_temperatures.ffill().loc[str(year)]
    average_temperatures.to_csv(
        f"{file_path_in.rsplit('/', 1)[0]}/{file_path_out}/average_temperatures.csv",
        sep=";",
    )
    all_temperatures.to_csv(
        f"{file_path_in.rsplit('/', 1)[0]}/{file_path_out}/all_temperatures.csv",
        sep=";",
    )


def read_and_filter_weather_datum(
    file: str, year: int, cols: List[str], error_value: float
) -> (float, pd.DataFrame):
    """Read a given file and filter values for given year"""
    data = pd.read_csv(file, sep=";", index_col=1)
    data.index = data.index.astype(str)
    stations_id = data["STATIONS_ID"].unique()
    create_datetime_index(data)
    filtered_data = data.loc[
        f"{year - 1}-12-30 00:00:00":f"{year}-12-31 23:00:00", cols
    ]
    full_filtered_data = pd.DataFrame(
        index=pd.date_range(
            start=f"{year - 1}-12-29 00:00:00",
            end=f"{year}-12-31 23:00:00",
            freq="H",
        ),
        data=filtered_data["TT_TU"],
    )
    replace_error_values(full_filtered_data, error_value)
    return stations_id, full_filtered_data


def create_datetime_index(filtered_data: pd.DataFrame):
    """Create a pd.DatetimeIndex to allow for resampling"""
    filtered_data["new_index"] = pd.to_datetime(
        filtered_data.index.str[:4]
        + "-"
        + filtered_data.index.str[4:6]
        + "-"
        + filtered_data.index.str[6:8]
        + " "
        + filtered_data.index.str[8:10]
        + ":00:00"
    )
    filtered_data.set_index("new_index", inplace=True)


def replace_error_values(filtered_data: pd.DataFrame, error_value: float):
    """Replace error value or empty value

    Use mean nearest neighbours if possible, else with median of given month"""
    false_idx_positions = [
        filtered_data.index.get_loc(val)
        for val in filtered_data.loc[
            filtered_data["TT_TU"] == error_value
        ].index
    ]
    false_idx_positions.extend(
        [
            filtered_data.index.get_loc(val)
            for val in filtered_data.loc[filtered_data["TT_TU"].isna()].index
        ]
    )
    monthly_medians = filtered_data.resample("M").median()
    for idx_pos in false_idx_positions:
        if (
            1 <= idx_pos <= len(filtered_data.index)
            and idx_pos + 1 not in false_idx_positions
            and idx_pos + 1 not in false_idx_positions
        ):
            filtered_data.iloc[idx_pos] = (
                filtered_data.iloc[idx_pos - 1]
                + filtered_data.iloc[idx_pos + 1]
            ) / 2
        else:
            filtered_data.iloc[idx_pos] = monthly_medians.loc[
                str(filtered_data.iloc[idx_pos].name)[:7]
            ]


def rolling_weighted_average(data: pd.DataFrame) -> List:
    """Calculate rolling weighted average using the last four data points.

    Solution modified starting from a ChatGPT solution
    """
    n = len(data)
    weighted_averages = [np.nan] * 3

    for i in range(3, n):
        weights = [0.05, 0.15, 0.3, 0.5]
        relevant_data = data[i - 3 : i + 1]
        weighted_avg = sum(w * d for w, d in zip(weights, relevant_data))
        weighted_averages.append(weighted_avg)

    return weighted_averages


if __name__ == "__main__":
    walk_data_folder_and_extract_mean(
        FILE_PATH_IN, FILE_PATH_OUT, YEAR, COLS, ERROR_VALUE
    )
