"""
Functions for the evaluation of technical demand response potentials.

Note:
    - For a dedicated description of the analysis, see jupyter notebook.
    - Another python script contains wrapper functions that make used of
      the functions here.

Notebook: DR_potential_evaluation.ipynb

@author: Johannes Kochems
"""
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_info(
    df, swapped_cols_dict, parameters, sector, years_dict, year, filter_sector
):
    """Extract info for (a) given parameter(s), sector and year from data

    Parameters
    ----------
    df : pandas.DataFrame
        grouped overall data set

    swapped_cols_dict : dict
        dict mapping cols to the parameter group of interest

    parameters : str or list
        parameter group to extract data for

    sector : str
        Sector to look for;
        possible values are "ind", "tcs" and "hoho"

    years_dict : dict
        dict holding all possible years

    year : str
        Year to extract data for; must be a key of years_dict;
        possible values are "SQ", "2020", "2025", "2030", "2035",
        "2040", "2045" and "2050"

    filter_sector : dict
        dict defining the sector filter

    Returns
    -------
    filtered_df : pandas.DataFrame
        Data set filtered for parameter(s) of interest
    """
    # Filter for sector, year and parameter and determine cols to use
    if isinstance(parameters, str):
        parameter = parameters
        parameters = list()
        parameters.append(parameter)
    elif not isinstance(parameters, list):
        raise TypeError(
            "parameters must be of type str (single parameter) "
            "or list (multiple parameters)."
        )

    filtered_df = df.loc[
        (df["Sektorenzuordnung"] == filter_sector[sector])
        & (df["Jahr"].isin(years_dict[year]))
    ]

    cols_to_use = []
    for parameter in parameters:
        cols_to_use.extend(
            [
                col
                for col in filtered_df.columns
                if col in swapped_cols_dict.keys()
                and swapped_cols_dict[col] == parameter
            ]
        )
    cols_to_use = list(set(cols_to_use))
    sources_cols = [col for col in cols_to_use if "Fundstelle" in col]

    # Filter for all nan rows and drop them
    filtered_df.set_index("study", append=True, inplace=True)
    nan_idx = filtered_df.loc[
        filtered_df[[col for col in cols_to_use if col not in sources_cols]]
        .isnull()
        .all(1)
    ].index
    filtered_df.drop(nan_idx, inplace=True)
    filtered_df.reset_index(level=1, drop=False, inplace=True)

    cols_to_use.extend(["Jahr", "Sektorenzuordnung", "study"])
    filtered_df = filtered_df[cols_to_use]

    return filtered_df


def add_value_counts_and_sources(filtered_df, drop_data_lack=True):
    """Add value counts to a filtered parameter data set

    Parameters
    ----------
    filtered_df : pandas.DataFrame
        Data set filtered for the parameter(s) of interest

    drop_data_lack : boolean
        If True, drop data sets where less than 3 data points exist

    Returns
    -------
    filtered_df : pandas.DataFrame
        Data set filtered for the parameter(s) of interest including
        information on the number of studies that contain values for this /
        these parameter(s) as well as the number of data points per
        process category
    """
    # Count number of publications by counting indices
    filtered_df["number_publications"] = filtered_df.index.value_counts()

    # Count not null entries for each study and aggregate by process category
    sources_cols = [col for col in filtered_df.columns if "Fundstelle" in col]
    filtered_df["number_entries"] = filtered_df.count(axis=1) - (
        len(sources_cols)
        + len(["Jahr", "Sektorenzuordnung", "study", "number_publications"])
    )
    entries = filtered_df["number_entries"].groupby(filtered_df.index).sum()
    filtered_df["abs_number_entries"] = entries
    filtered_df.drop(columns=["number_entries"], inplace=True)

    if drop_data_lack:
        filtered_df = filtered_df.loc[filtered_df["abs_number_entries"] >= 3]

    # Join all the sources string information by process category
    for col in sources_cols:
        entries = (
            filtered_df[col]
            .groupby(filtered_df.index)
            .apply(lambda x: "; ".join(x))
        )
        filtered_df["all_sources_" + col] = entries

    return filtered_df


def groupby_process_category(filtered_df, agg_rule="mean"):
    """Group the filtered parameter data set by process categories

    Parameters
    ----------
    filtered_df : pandas.DataFrame
        Data set filtered for the parameter(s) of interest

    agg_rule : str
        Aggregation rule to apply; defaults to "mean"
        Since the parameters come with distinct columns, it actually has
        no effect. "mean" is preferred over "sum" since "sum" creates 0
        entries when aggregating only np.nan values

    Returns
    -------
    grouped_df : pandas.DataFrame
        Data set for parameter(s) of interested grouped by
        process categories
    """
    agg_rules = {
        col: agg_rule
        for col in filtered_df.columns
        if "all_sources" not in col
        and "Fundstelle" not in col
        and col not in ["Jahr", "Sektorenzuordnung", "study"]
    }
    for col in filtered_df.columns:
        if "all_sources" in col:
            agg_rules[col] = lambda x: "; ".join(x)

    grouped_df = filtered_df.groupby(filtered_df.index).agg(agg_rules)

    return grouped_df


def transpose_and_split(grouped_df):
    """Transpose and split the grouped data set

    The DataFrame fed in is transposed and split into the actual
    numeric parameter data, the counts of the number of publications
    and data points as well as the combined sources information

    Parameters
    ----------
    grouped_df : pandas.DataFrame
        Data set for parameter(s) of interested grouped by
        process categories

    Returns
    -------
    numeric_df : pandas.DataFrame
        Data set containing the actual numeric parameter information

    count_df : pandas.DataFrame
        Data set containing the counts on the number of publications that
        contain information as well as the number of data points per process
        category

    all_sources_df : pandas.DataFrame
        Data set containing the combined sources information for
        the parameter
    """
    transposed_df = grouped_df.T

    all_sources_rows = [
        row for row in grouped_df.columns if "all_sources" in row
    ]
    count_rows = ["number_publications", "abs_number_entries"]
    numeric_rows = [
        row
        for row in grouped_df.columns
        if row not in all_sources_rows and row not in count_rows
    ]

    numeric_df = transposed_df.loc[numeric_rows]
    numeric_df = numeric_df.astype("float64")
    count_df = transposed_df.loc[count_rows]
    all_sources_df = transposed_df.loc[all_sources_rows]

    return numeric_df, count_df, all_sources_df


def extract_sample_stats(
    numeric_df,
    save=False,
    path_folder="./out/stats/",
    file_name="stats.csv",
):
    """Extract the sample stats for the parameter data set

    Calculate statistical moments using the built-in
    pandas.DataFrame.describe() method and add quantile 5%, 10%,
    90% and 95% to it.

    Parameters
    ----------
    numeric_df : pandas.DataFrame
        Data set containing the actual numeric parameter information

    save : boolean
        Save the extracted parameter statistics if True

    path_folder : str
        Path where to store the parameter statistics

    file_name : str
        File name to use for storing the parameter statistics

    Returns
    -------
    stats_df : pandas.DataFrame
        Data set containing statistics for the parameter(s)
    """
    if not numeric_df.empty:
        stats_df = numeric_df.describe()
        quantiles = {"5%": 0.05, "10%": 0.1, "90%": 0.9, "95%": 0.95}

        for key, val in quantiles.items():
            stats_df.loc[key] = numeric_df.quantile(val).values

        if save:
            stats_df.to_csv(path_folder + file_name, sep=";", decimal=".")

        return stats_df


def extract_sample_sources(
    all_sources_df,
    save=False,
    path_folder="./out/sources/",
    file_name="sources.csv",
    return_sources=False,
):
    """Extract the sources for the parameter data set

    Parameters
    ----------
    all_sources_df : pandas.DataFrame
        Data set containing the combined sources information
        per pocess category

    save : boolean
        Save the combined sources information if True

    path_folder : str
        Path where to store the combined sources

    file_name : str
        File name to use for storing the combined sources

    return_sources : boolean
        If True, return the unique sources string

    Returns
    -------
    unique_sources_string : str
        Combined sources string
    """
    if not all_sources_df.empty:
        combined_sources = all_sources_df.agg(lambda x: "; ".join(x))
        combined_sources = combined_sources.values + "; "
        combined_sources_string = combined_sources.sum()

        unique_sources_set = set(combined_sources_string.split("; "))
        unique_sources_set.discard("")

        unique_sources_string = "; ".join(unique_sources_set)
        unique_sources_string = unique_sources_string.replace(".0", "")
        unique_sources_string = unique_sources_string.replace(": nan", "")

        if save:
            with open(path_folder + file_name, "w", encoding="UTF8") as opf:
                opf.write(unique_sources_string)

        if return_sources:
            return unique_sources_string


def create_boxplot(
    numeric_df,
    counts_df,
    title,
    ylabel,
    colors=None,
    year="SQ",
    ylim=[0, 3000],
    use_colors=False,
    use_limits=True,
    swarmplot=False,
    savefig=False,
    show_title=True,
    include_year=True,
    include_line_break=False,
    path_folder="./out/plots/",
    file_name="parameter",
):
    """Creates a boxplot for relevant parameter(s) by process categories

    Parameters
    ----------
    numeric_df : pandas.DataFrame
        Data set containing the actual numeric parameter information

    counts_df : pandas.DataFrame
        Data set containing the counts on the number of publications that
        contain information as well as the number of data points per process
        category

    year : str
        Year for which the plot shall be created

    title : str
        Title for the plot

    ylabel : str
        label for the yaxis

    colors : pd.DataFrame
        pd.DataFrame containing the category / color mapping

    ylim : list
        limits for the yaxis

    use_colors : boolean
        If True, colors from the given colors DataFrame will be used, default
        ones elsewhise

    use_limits : boolean
        If True, yaxis limits will be obtained from the extreme values of
        the data including some space

    swarmplot : boolean
        Create an overlay bee swarm plot if True using seaborn

    savefig : boolean
        Save figure to png if True

    show_title : boolean
        Show / hide the given title

    include_year : boolean
        Include year in the title if True

    include_line_break : boolean
        If True, add line break in xaxis label

    path_folder : str
        Path where the png file shall be stored

    file_name : str
        Name of the .png file to be stored
    """
    # Terminate execution when DataFrame is entirely empty
    if numeric_df.empty:
        warnings.warn("No numeric data to plot.", UserWarning)
        return None

    fig, ax = plt.subplots(figsize=(15, 5))

    connection = "; "
    if include_line_break:
        connection = "\n"

    numeric_df_plot = numeric_df.rename(
        columns={
            col: col
            + connection
            + "n: "
            + str(int(counts_df.at["number_publications", col]))
            + ", m: "
            + str(int(counts_df.at["abs_number_entries", col]))
            for col in counts_df.columns
        }
    )

    if not swarmplot:
        _ = numeric_df_plot.plot(kind="box", ax=ax)
    else:
        if use_colors:
            _ = sns.boxplot(
                data=numeric_df_plot,
                ax=ax,
                width=0.5,
                boxprops=dict(alpha=0.2),
                palette=sns.color_palette(
                    [
                        el
                        for el in colors.loc[
                            numeric_df.columns, "Farbe (matplotlib strings)"
                        ].values
                    ]
                ),
            )
            _ = sns.swarmplot(
                data=numeric_df_plot,
                ax=ax,
                palette=sns.color_palette(
                    [
                        el
                        for el in colors.loc[
                            numeric_df.columns, "Farbe (matplotlib strings)"
                        ].values
                    ]
                ),
            )
        else:
            _ = sns.boxplot(
                data=numeric_df_plot,
                ax=ax,
                width=0.5,
                boxprops=dict(alpha=0.2),
            )
            _ = sns.swarmplot(data=numeric_df_plot, ax=ax)

    if show_title:
        if include_year:
            if year == "SQ":
                _ = plt.title(title + " im Status quo")

            else:
                _ = plt.title(title + " im Jahr " + year)
        else:
            _ = plt.title(title)

    if use_limits:
        minimum = numeric_df.min().min()
        maximum = numeric_df.max().max()

        if minimum >= 0:
            ylim = [minimum - 0.1 * minimum, maximum + 0.1 * maximum]
        else:
            ylim = [minimum + 0.1 * minimum, maximum - 0.1 * maximum]

    _ = plt.ylim(ylim)
    _ = plt.xlabel("Lastmanagementkategorie", labelpad=10)
    _ = plt.ylabel(ylabel, labelpad=10)
    _ = plt.xticks(rotation=90)

    if savefig:
        plt.savefig(
            path_folder + file_name + "_boxplot.png",
            dpi=150,
            bbox_inches="tight",
        )

    _ = plt.show()
    plt.close()


def get_nlargest(stats_df, metric="50%", n=5):
    """Get the process categories with the largest potentials

    Parameters
    ----------
    stats_df: pandas.DataFrame
        Data set containing statistics for the parameter(s)

    metric: str
        Metric which shall be used for detecting the n largest processes

    n: int
        Determines how many values shall be used (n largest)

    Returns
    -------
    process_list: list
        List containing the processes with the largest potentials
    """
    process_list = []
    if not stats_df.empty:
        process_list = list(
            stats_df.loc[metric].sort_values(ascending=False).index.values[:n]
        )

    return process_list
