import pandas as pd
import os

from drpotentials.tools import make_directory_if_missing

FILE_PATH_IN = "D:/tubCloud2/Promotion/Literatur/Lastmanagement/full_download_dsmlib_region4FLEX/full_download_dsmlib_region4FLEX/results"  # noqa: E501
FILE_PATH_OUT = "D:/tubCloud2/Promotion/Literatur/Lastmanagement/full_download_dsmlib_region4FLEX/prepared/"


def reaggregate_parameter(root_path: str, year: int):
    """Reaggregate piece of information across all German districts"""
    envelopes_root_path = f"{root_path}/{year}/envelopes/"
    parameters = pd.DataFrame()
    for path in os.listdir(envelopes_root_path):
        sub_path = os.path.join(envelopes_root_path, path)
        if os.path.isdir(sub_path):
            cluster_data_set = pd.DataFrame()
            for file in os.listdir(sub_path):
                deaggregated = pd.read_csv(f"{sub_path}/{file}", index_col=0)
                deaggregated = deaggregated.sum(axis=1)
                cluster = file.split("_", 2)[-1].rsplit("_", 2)[0]
                value = file[:5]
                cluster_data_set[value] = deaggregated
                make_directory_if_missing(f"{FILE_PATH_OUT}/{year}")
                deaggregated.to_csv(f"{FILE_PATH_OUT}/{year}/{file}", sep=";")
            calculate_metrics(cluster, cluster_data_set, parameters)
    parameters.to_csv(f"{FILE_PATH_OUT}/{year}/parameters.csv", sep=";")


def calculate_metrics(
    cluster: str, data_set: pd.DataFrame, parameters: pd.DataFrame
):
    """Calculate potential parameters for cluster data set"""
    parameters.at[cluster, "max_capacity_in_MW"] = (
        data_set["p_set"].max().item()
    )
    parameters.at[cluster, "min_capacity_in_MW"] = (
        data_set["p_set"].min().item()
    )
    parameters.at[cluster, "power_consumption_in_MWh"] = data_set[
        "p_set"
    ].sum()
    parameters.at[cluster, "negative_potential_max_in_MW"] = data_set[
        "p_max"
    ].max()
    parameters.at[cluster, "negative_potential_ave_in_MW"] = data_set[
        "p_max"
    ].mean()
    parameters.at[cluster, "negative_potential_min_in_MW"] = data_set[
        "p_max"
    ].min()
    parameters.at[cluster, "positive_potential_max_in_MW"] = -data_set[
        "p_min"
    ].min()
    parameters.at[cluster, "positive_potential_ave_in_MW"] = -data_set[
        "p_min"
    ].mean()
    parameters.at[cluster, "positive_potential_min_in_MW"] = -data_set[
        "p_min"
    ].max()
    parameters.at[cluster, "maximum_allowed_capacity_in_MW"] = (
        data_set["p_set"] + data_set["p_max"]
    ).max()


if __name__ == "__main__":
    reaggregate_parameter(FILE_PATH_IN, 2018)
    reaggregate_parameter(FILE_PATH_IN, 2030)
