import pandas as pd
import os

FILE_PATH_IN = "D:/tubCloud2/Promotion/Literatur/Lastmanagement/full_download_dsmlib_region4FLEX/full_download_dsmlib_region4FLEX/results"  # noqa: E501
FILE_PATH_OUT = "D:/tubCloud2/Promotion/Literatur/Lastmanagement/full_download_dsmlib_region4FLEX/prepared/"


def reaggregate_parameter(root_path: str, year: int):
    """Reaggregate piece of information across all German districts"""
    envelopes_root_path = f"{root_path}/{year}/envelopes/"
    for path in os.listdir(envelopes_root_path):
        sub_path = os.path.join(envelopes_root_path, path)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                deaggregated = pd.read_csv(f"{sub_path}/{file}", index_col=0)
                deaggregated = deaggregated.sum(axis=1)
                make_directory_if_missing(f"{FILE_PATH_OUT}/{year}")
                deaggregated.to_csv(
                    f"{FILE_PATH_OUT}/{year}/{file}", sep=";"
                )


def make_directory_if_missing(folder: str) -> None:
    """Add directories if missing (solution created querying ChatGPT)"""
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    except OSError as e:
        print(f"Failed to create directory: {e}")


if __name__ == "__main__":
    reaggregate_parameter(FILE_PATH_IN, 2018)
    reaggregate_parameter(FILE_PATH_IN, 2030)
