# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:05:59 2020

This file contains some function definitions for the notebook
used for clustering of technical demand response potentials.

Note:
    - Method definitions here are tailored to the usage in the notebook and
      therefore not written as universally applicable (e.g. containing no Error
      handling).
    - Function group_potential is similar to group_transformers from the module
      functions_for_data_preparation.py used for the ER fundamental power
      market model but a bit more generic.

Notebook: DR_potential_clustering_VXX.ipynb

@author: jkochems
"""

import numpy as np
import pandas as pd


def create_parameter_combinations(parameters_list, cols):
    """Function creates combinations of measures of central tendency,
    demand response parameters as well as a given sector.

    Parameters
    ----------
    parameters_list: list
        The parameters for which the combinations shall be determined

    cols: list
        The measures of central tendency to be used

    sector: str
        The sector for which the combinations shall be determined

    Returns
    -------
    combinations_list: list
        The list of all possible combinations
    """
    combinations_list = []

    # Include all combinations in the list (i.e. measures of central tendencies
    # and parameters)
    col, par = np.meshgrid(cols, parameters_list)
    tuples_list = list(zip(col.flatten(), par.flatten()))
    for tup in tuples_list:
        combinations_list.append(tup[0] + "_" + tup[1])

    return combinations_list


def group_potential(
    df,
    grouping_cols,
    mean_cols=[],
    sum_cols=[],
    min_cols=[],
    max_cols=[],
    other_cols=[],
    sector=None,
    drop=[],
    add_cluster=True,
    weighted_ave=True,
    weight=[],
):
    """Function does a grouping using the aggregation functions specified.
    It usually is applied after determining clusters within a cluster analysis.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be grouped

    grouping_cols: list
        The column(s) to use for the grouping

    mean_cols: list
        Column(s) for which the weigthed average shall be applied

    sum_cols: list
        Column(s) for which the sum shall be calculated

    sector: str
        The sector (one of "ind", "tcs" and "hoho")

    drop: list
        The column(s) to be dropped in grouping

    add_cluster: boolean
        If True, add sector and cluster information as index

    weighted_ave: boolean
        Determines whether to use a weighted average or a simple one

    weight: str
        Column to use for calculating weigthed averages

    Returns
    -------
    df_grouped: pd.DataFrame
        The grouped DataFrame
    """

    # Ensure that data is available
    sum_cols = [el for el in sum_cols if el in df.columns]
    mean_cols = [el for el in mean_cols if el in df.columns]
    min_cols = [el for el in min_cols if el in df.columns]
    max_cols = [el for el in max_cols if el in df.columns]
    other_cols = [el for el in other_cols if el in df.columns]

    if not len(mean_cols) == 0:
        if weighted_ave:
            # Calculate a weighted average value using a lambda function
            wm = lambda x: np.average(x, weights=df.loc[x.index, weight])
            df_grouped_mean = (
                df[mean_cols + grouping_cols].groupby(grouping_cols).aggregate(wm)
            )
        else:
            df_grouped_mean = (
                df[mean_cols + grouping_cols].groupby(grouping_cols).mean()
            )
    else:
        df_grouped_mean = df[mean_cols + grouping_cols].groupby(grouping_cols).nth(0)

    #  Sum up the values for the respective columns
    df_grouped_sum = df[sum_cols + grouping_cols].groupby(grouping_cols).sum()

    # Take minimum resp. maximum values as aggregation values
    df_grouped_min = df[min_cols + grouping_cols].groupby(grouping_cols).min()
    df_grouped_max = df[max_cols + grouping_cols].groupby(grouping_cols).max()

    # Use nth(0), i.e. first value for the respective columns with only one entry
    df_grouped_other = df[other_cols + grouping_cols].groupby(grouping_cols).nth(0)

    # Combine the different subsets again to obtain a single DataFrame
    # and set index to include cluster number which will be dropped
    df_grouped = pd.concat(
        [
            df_grouped_mean,
            df_grouped_sum,
            df_grouped_other,
            df_grouped_min,
            df_grouped_max,
        ],
        axis=1,
        join="inner",
    ).reset_index()

    if add_cluster:
        new_index = sector + "_cluster-" + df_grouped["cluster"].apply(str)
        df_grouped = df_grouped.set_index(new_index).drop(["cluster"], axis=1)

    df_grouped = df_grouped.drop(
        [col for col in df_grouped.columns for el in drop if el in col], axis=1
    )

    return df_grouped


def write_multiple_sheets(sector_dict, path_folder, filename):
    """
    Function writes all DataFrames contained in a dict to an Excel file.
    """
    writer = pd.ExcelWriter(path_folder + filename, engine="xlsxwriter")

    for k, v in sector_dict.items():
        v.to_excel(writer, sheet_name=k)

    writer.save()


def map_column_names(availability_time_series, availability_categories):
    """Map column name to the potential information and adapt column names

    Parameters
    ----------
    availability_categories : list of tuples
        categories to be used

    availability_time_series : pd.DataFrame
        availability time series DataFrame

    Returns
    -------
    availability_time_series : pd.DataFrame
        availability time series DataFrame with renamed columns
    """
    availability_time_series = availability_time_series.rename(
        columns=availability_categories
    )
    multi_index = pd.MultiIndex.from_tuples(
        availability_time_series.columns.values
    ).set_names(["Prozesskategorie", "sector"])
    availability_time_series.columns = multi_index

    return availability_time_series


def determine_missing_cols(process_categories, availability_time_series,
                           sector=None, kind=None):
    """Print info on which columns are missing

    Parameters
    ----------
    process_categories : pd.MultiIndex
        all process categories (tuple of category and sector)

    availability_time_series : pd.DataFrame
        availability time series DataFrame

    sector : str
        sector (one of "ind", "tcs", "hoho")

    kind : str
        kind of potential ("pos" for positive or "neg" for negative)
    """
    pot_cols = set(
        process_categories[
            process_categories.get_level_values(1) == sector
            ])
    ava_cols = set(availability_time_series.columns.to_list())
    diff_cols = list(pot_cols - ava_cols)

    # print info on which columns are missing
    print(f"Missing columns for {sector} in {kind}itive direction:")
    print(40 * "-")
    print(diff_cols)
    print()


def assign_periodical_values(periods, periodical_factors_dict):
    """Assigns periodical (hourly, weekly or monthly) values

    Parameters
    ----------
    periods : range
        Range object holding the range of hours, days or months

    periodical_factors_dict : dict
        Dictionary indexed by process categories containing the assigned
        hourly, weekly or monthly availability factors

    Returns
    -------
    factors_pos : dict
        Positive availability factors

    factors_neg : dict
        Negative availability factors
    """
    factors_pos = dict()
    factors_neg = dict()

    for key, value in periodical_factors_dict.items():
        factors_pos[key] = dict(zip(periods, value["pos"]))
        factors_neg[key] = dict(zip(periods, value["neg"]))

    return factors_pos, factors_neg


def create_synthetic_profile_factors(periods_factors):
    """Create synthetic profiles based on hourly, daily and monthly patterns

    Parameters
    ----------
    periods_factors : dict
        Dictionary mapping the periods (hours, days or months to the respective
        factors)

    Returns
    -------
    availability_factors : dict
        Nested dictionary indexed by sector and direction ("pos" or "neg")
    """
    availability_factors = dict()

    for period, factors in periods_factors.items():
        period_name = factors[0]
        period_factors = factors[1]
        pos_factors, neg_factors = assign_periodical_values(period,
                                                            period_factors)
        availability_factors[(period_name, "pos")] = pos_factors
        availability_factors[(period_name, "neg")] = neg_factors

    return availability_factors


def assign_availability_remaining(
    params,
    availability_time_series,
    synthetic_cols,
    factors,
    sector,
    kind
):
    """Assign availability time series for the remaining categories

    Appends the newly created columns to the parameter DataFrame inplace

    Parameters
    ----------
    params : pd.DataFrame
        Parameter data set

    availability_time_series : pd.DataFrame
        availability time series DataFrame

    synthetic_cols : list
        Categories for which synthetic load profiles shall be created

    factors: dict
        A nested dict containing hourly, weekly and monthly patterns from which
        the synthetic profiles are build of

    sector: str
        Sector for which availabilities shall be determined

    kind: str
        Direction of potentials ("pos" for positive or "neg for negative)
    """
    processes = set(params.index[params.index.get_level_values(1) == sector])
    availabilities = set(availability_time_series.columns.to_list())
    diff_cols = list(processes - availabilities)

    # By default, assign all columns a constant availability
    scalar_dict = {c: 1 for c in diff_cols if c not in synthetic_cols}
    if not len(scalar_dict) == 0:
        print(scalar_dict)
        availability_time_series[diff_cols] = pd.DataFrame(
            scalar_dict, index=availability_time_series.index
        )

    for col in synthetic_cols:
        hours = availability_time_series.index.hour.map(
            factors[("hours", kind)][col]
        )
        days = availability_time_series.index.dayofweek.map(
            factors[("days", kind)][col]
        )
        months = availability_time_series.index.month.map(
            factors[("months", kind)][col]
        )

        availability_time_series[col] = pd.Series(
            hours * days * months, index=availability_time_series.index
        )


# Function definitions taken from this stackoverflow issue:
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
# accessed 27.08.2020
def get_redundant_pairs(df):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=None, threshold=None):
    """Obtain n largest correlations or largest correlations up to threshold"""
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    if n is not None and threshold is None:
        to_return = au_corr[0:n]
    elif threshold is not None and n is None:
        to_return = au_corr[au_corr >= threshold]
    else:
        raise ValueError("Either specify 'n' XOR 'threshold'")
    return to_return
