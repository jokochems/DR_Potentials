# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:53:04 2020

This file contains some function definitions for the notebook
used for evaluation of technical demand response potentials.

Note: 
    - Method definitions here are tailored to the usage in the notebook and
      therefore not written as universally applicable (e.g. containing no Error 
      handling).
    - Methods create_info_boxplots and create_projection_boxplot do more or
      less the same. It would be nicer to only use one function definition
      and to separate the data preparation from the actual plotting routine.

Notebook: DR_potential_evaluation_VXX.ipynb

@author: jkochems
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter


def extract_info(dict_origin, cols_info, drop=True, sample_stats=False,
                 add_sources=False, drop_datalack=True,
                 modify_categories=False, agg_rule="sum", weights=None):
    """ 
    Function extracts information on a specific demand response parameter
    from all publications in the given dictionary. It creates a further 
    dictionary to store the information and concatenates the DataFrames 
    with the columns of interest after that.
    
    It also stores the number of publications that contain info on a specific
    parameter per process as well as the amount of single parameter values
    that could be used for the further evaluation.
    
    Finally, it collects and rearranges sources information for the information
    on the demand response parameter of interest.
    
    Usage: 
        - Set drop to True if you want to have comparable plots (all processes
          listed) which then include empty categories
        - Set drop_datalack to True if you want to exclude processes for which
          less than 3 data points exist and hence no boxplot can be drawn
        - Set sample_stats to True if you wish to include information on the
          sample size. Since this introduces computational expensive operations,
          it is set to False by default. The idea is to run it once at the end, 
          when a final data analysis set is needed.
        - Set add_sources to True if you wish to include and rearrange sources
          information. Since this introduces computational expensive operations,
          it is set to False by default. The idea is to run it once at the end, 
          when a final data analysis set is needed.
        - Set modify_categories to True if you wish to use standardized categories
          from the method comparison instead of the given ones.
    
    Parameters
    ----------    
    dict_origin: dict
        Dictionary from which the information is obtained
    
    cols_info: list of str 
        columns to be evaluated on the parameter of interest
            
    drop: boolean
        If True, empty columns will be dropped
    
    sample_stats: boolean
        Obtain info on the sample size
        
    add_sources: boolean
        If True, information on sources (page areas) is included
        
    drop_datalack: boolean
        If True, processes with less than 3 data points will be dropped
        
    modify_categories: boolean
        If True, a self-defined set of processes / demand response categories
        is used and the original values are combined together using a given
        aggregation function
        
    agg_rule: str
        the aggregation function to be applied on the columns ("sum" resp.
        "weighted_average")
    
    Returns
    -------
    dict_info: dict
        Dictionary containing the information of interest for every publication
    
    df_info: pd.DataFrame
        pd.DataFrame containing the information on the parameter of interest 
        across all publications
    
    """

    # Create a dict containing the info on a given parameter 
    # per publication and year
    dict_info = {}
    process_cols = ["Prozess"]
    # Create a copy of the columns in order not to manipulate the original input
    cols_to_use = cols_info.copy()

    # modify_categories changes the original categories and takes the one of "Prozesskategorie 1"
    # except for cases where replacement mode equals to -1
    if modify_categories:
        cols_to_use.extend(["Prozesskategorie 1", "replacement_mode"])

        if agg_rule == "weighted_average":
            # Remove columns "Prozess" and "year_key" from weights because otherwise,
            # these would be duplicated
            duplicates = ["Prozess", "year_key"]
            weights = [el for el in weights if "Fundstelle" not in el and el not in duplicates]
            cols_to_use.extend(weights)

    process_cols.extend(["year_key"])

    if not add_sources:
        cols_to_use = [el for el in cols_to_use if not "Fundstelle" in el]

    for k, v in dict_origin.items():
        cols = [el for el in cols_to_use if el in v.columns]
        dict_info[k] = v[cols]

        # Do the actual modification of categories and aggregate data (if necessary)
        if modify_categories:
            # Replace values for "Prozess" column with those from "Prozesskategorie 1" column
            # if not the original names shall be kept
            dict_info[k].loc[dict_info[k]["replacement_mode"] != -1,
                             "Prozess"] = dict_info[k].loc[dict_info[k]["replacement_mode"] != -1,
                                                           "Prozesskategorie 1"]
            dict_info[k].loc[dict_info[k]["replacement_mode"] == -1, "replacement_mode"] = 0

            # Create a sub DataFrame with duplicate values for "Prozesskategorie 1"
            # Keep all entries in the first place in order to later apply
            # aggregation functions on them and drop them from the original DataFrame
            sub_df = dict_info[k].loc[dict_info[k]["Prozesskategorie 1"].duplicated(keep=False)]
            dict_info[k] = dict_info[k].drop(sub_df.index)

            # Define on which columns of the sub DataFrame to perform (no) aggregation
            # For object columns the 0th value is taken because values are the same
            # (this may not hold for sources information, but is neglected here)
            # For other columns, sum or weighted average apply as aggregation functions
            # After that, group DataFrames by "Prozess" column and combine them again
            no_agg_cols = [col for col in cols
                           if dict_info[k][col].dtype == object or "Fundstelle" in col]
            agg_cols = [col for col in cols
                        if col not in no_agg_cols or col == "Prozess"]

            sub_df_no_agg = sub_df.loc[:, [col for col in cols
                                           if col in no_agg_cols]].groupby("Prozess").nth(0)

            # Weigths used holds the weighting cols used for calculating a weigthed average
            # which have to be removed again
            weights_used = []

            if not agg_rule == "weighted_average":
                # Standard aggregation rule is calculating the sum
                sub_df_agg = sub_df.loc[:, [col for col in cols
                                            if col in agg_cols]].groupby("Prozess").sum()

            else:
                weights_used = [el for el in cols
                                if el in weights and el not in ["Prozess", "year_key"]]
                # If multiple weight_cols are given, transform into one (using median)
                sub_df["weights"] = sub_df[weights_used].median(axis=1)
                sub_df = sub_df.fillna(sub_df.median(axis=0))

                # Introduce a lambda function for calculating a weighted average
                if not sub_df["weights"].sum() == 0:
                    wm = lambda x: np.average(x, weights=sub_df.loc[x.index, "weights"], axis=0)
                    sub_df_agg = sub_df.loc[:, [col for col in cols
                                                if col in agg_cols]].groupby("Prozess").aggregate(wm)
                else:
                    sub_df_agg = sub_df.loc[:, [col for col in cols
                                                if col in agg_cols]].groupby("Prozess").mean()

            sub_df = pd.concat([sub_df_no_agg, sub_df_agg], axis=1).reset_index(drop=False)

            # Combine the manipulated sub DataFrame with the original one
            # (without the aggregated rows) again
            dict_info[k] = pd.concat([dict_info[k], sub_df], axis=0, sort=False).drop(
                ["Prozesskategorie 1", "replacement_mode"], axis=1).set_index(process_cols)
            # Drop weight columns with information not needed anymore and remove empty columns
            dict_info[k] = dict_info[k].drop(weights_used, axis=1).dropna(how="all", axis=1)

        # No modification on categories (take the original ones)
        else:
            # Remove the number of columns by dropping those only containing NaN entries
            dict_info[k] = v[cols].set_index(process_cols).dropna(how="all", axis=1)

        if sample_stats:
            dict_info[k]["info_available"] = \
                dict_info[k].loc[:, [el for el in dict_info[k].columns
                                     if "Fundstelle" not in el]].notnull().any(axis=1).astype(int)

        if add_sources:
            source_col = [el for el in dict_info[k].columns
                          if "Fundstelle" in el]
            dict_info[k][source_col] = k.split("_")[0] + \
                                       ", S. " + dict_info[k][source_col].astype(str) + "; "

    # Merge all infos on a given parameter together 
    # for a given process and year across all publications
    df_info = pd.DataFrame(columns=process_cols)

    for k, v in dict_info.items():
        try:
            df_info = pd.merge(df_info, v, on=process_cols, suffixes=["", "_" + k], how="outer")
        except ValueError as err:
            print(err)
            print(k + " failed!")

    # Add info on the number of publications as well as the number of data values
    # drop redundant info not needed anymore
    if sample_stats:
        df_info["publications_number"] = df_info.filter(regex="info_available").sum(axis=1)
        df_info = df_info.drop([col for col in df_info.columns if "info_available" in col], axis=1)

        df_info["number_entries"] = df_info.count(axis="columns", numeric_only=True) - 1

        if drop:
            df_info = df_info[(df_info["publications_number"] != 0)
                              & (df_info["number_entries"] != 0)]

            # Drop categories with 3 or less entries
        if drop_datalack:
            df_info = df_info[df_info["number_entries"] >= 3]

    # Drop empty columns or columns containing 3 or less entries (if not sample_stats)    
    else:
        if drop:
            cols_to_ignore = process_cols
            df_info = df_info.dropna(how="all", subset=[el for el in df_info.columns
                                                        if not el in cols_to_ignore],
                                     axis=0)

        if drop_datalack:
            df_info = df_info[df_info.count(axis="columns", numeric_only=True) >= 3]

    # Concatenate together all sources information and delete source cols not needed anymore
    if add_sources:
        source_cols = [el for el in df_info.columns if "Fundstelle" in el]
        df_info["sources"] = df_info[source_cols].fillna("").sum(axis=1)
        df_info = df_info.drop(source_cols, axis=1)

    # Do not include year_key into index, but only use process column(s)
    df_info = df_info.set_index(process_cols[:-1])

    return dict_info, df_info


def create_info_dict_for_years(df_info, years_dict, negate=False,
                               sample_stats=False, add_sources=False):
    """ 
    Function takes the extracted information on a specific demand response 
    parameter. It creates dictionaries that store the data for every year
    and transforms its shape such that it can be visualized using a boxplot
    in the following.
    
    Parameters
    ----------
    df_info: pd.DataFrame or list of pd.DataFrames
        pd.DataFrame(s) containing the information on the parameter of interest 
        across all publications (obtained from method extract info)
    
    years_dict: dict
        Dictionary of years resp. year groups to be evaluated
    
    negate: boolean
        if True the values are negated, i.e. multiplied with -1 (for negative
        potential information)
    
    sample_stats: boolean
        True if info on the sample size is included
        
    add_sources: boolean
        True if information on sources (page areas) is included
    
    Returns
    -------
    dict_info_years: dict or list of dicts
        Dictionary resp. dictionaries to store the info on the relevant 
        parameter, indexed by years
        
    dict_sources: dict or list of dicts
        Dictionary resp. dictionaries to store the sources info on the relevant 
        parameter, indexed by years
        
    """

    if not isinstance(df_info, list):

        dict_info_years = {}
        dict_sources = {}
        rows_to_drop = ["year_key"]

        # Split up sources information and actual data
        if "sources" in df_info.columns:
            rows_to_drop.append("sources")

        for year in years_dict.keys():
            dict_info_years[year] = df_info[df_info["year_key"] == year].T.drop(
                rows_to_drop, axis="rows").astype(float)
            if "sources" in df_info.columns:
                dict_sources[year] = df_info[df_info["year_key"] == year]["sources"].unique().sum()

            if negate:
                rows_to_ignore = []

                if sample_stats:
                    rows_to_ignore.extend(["publications_number", "number_entries"])

                # Workaround: to ignore publications number and number entries 
                # in negation, simply multiply with -1 twice
                dict_info_years[year] = dict_info_years[year].mul(-1)
                dict_info_years[year].loc[rows_to_ignore] = dict_info_years[year].loc[rows_to_ignore].mul(-1)

    else:
        dict_info_years = []
        dict_sources = []

        for el in df_info:
            new_info, new_source = create_info_dict_for_years(el, years_dict,
                                                              negate=negate,
                                                              sample_stats=sample_stats,
                                                              add_sources=add_sources)
            dict_info_years.append(new_info)
            dict_sources.append(new_source)

    return dict_info_years, dict_sources


def save_info_stats(dict_info_years, path_folder, filename, sample_stats=False,
                    add_sources=False):
    """
    Function takes the dictionary containing info on the relevant parameter
    indexed by years and calculates statistics for every DataFrame in the dict.
    These statistics are then written to an Excel File.
    
    Parameters
    ----------
    dict_info_years: dict or list of dicts
        Dictionary resp. dictionaries to store the info on the relevant 
        parameter, indexed by years
    
    path_folder: str
        Path where the Excel File shall be stored
    
    filename: str
        Name of the Excel File to be stored    
        
    sample_stats: boolean
        True if info on the sample size is included
    
    add_sources: boolean
        True if information on sources (page areas) is included 
        
    """

    writer = pd.ExcelWriter(path_folder + filename, engine="xlsxwriter")

    rows_to_ignore = []

    for k, v in dict_info_years.items():

        if sample_stats:
            rows_to_ignore.extend(["number_entries", "publications_number"])
        if add_sources and "sources" in v.columns:
            rows_to_ignore.append("sources")

        if not v.empty:
            v_info = v.drop(rows_to_ignore, axis=0)
            v_info.describe().to_excel(writer, sheet_name=k)

    writer.save()


def save_sources(dict_sources, path_folder, filename):
    """
    Function to write the sources information into excel sheets.
    """
    writer = xlsxwriter.Workbook(path_folder + filename)

    for k, v in dict_sources.items():
        ws = writer.add_worksheet(k)
        ws.write(0, 0, v)

    writer.close()


def create_info_boxplot(dict_info_years, year, title, ylabel, colors, ylim=[0, 3000],
                        use_colors=False, use_limits=True, sample_stats=False, add_sources=False,
                        swarmplot=False, savefig=False, show_title=True, path_folder="./out/",
                        filename="DR_parameter"):
    """
    Function creates a boxplot for the relevant parameter, indexed by
    processes.
    
    Parameters
    ----------
    dict_info_years: dict or list of dicts
        Dictionary resp. dictionaries to store the info on the relevant 
        parameter, indexed by years
        
    year: str
        Year for which the plot shall be created
        
    title: str
        Title for the plot
    
    ylabel: str
        label for the yaxis
        
    colors: pd.DataFrame
        pd.DataFrame containing the category / color mapping
        
    ylim: list
        limits for the yaxis
        
    use_colors: boolean
        If True, colors from the given colors DataFrame will be used, default
        ones elsewhise

    use_limits: boolean
        If True, yaxis limits will be obtained from the extreme values of
        the data including some space
        
    sample_stats: boolean
        True if info on the sample size is included
    
    add_sources: boolean
        True if information on sources (page areas) is included
        
    swarmplot: boolean
        Create an overlay bee swarm plot if True using seaborn

    savefig: boolean
        Save figure to png if True
        
    show_title: boolean
        Show / hide the given title
        
    path_folder: str
        Path where the png file shall be stored
    
    filename: str
        Name of the png file to be stored    
    """

    to_plot = dict_info_years[year].copy()

    # Terminate execution when DataFrame is entirely empty which
    # might be because data has been dropped because there was too few.
    if to_plot.empty:
        return None

    fig, ax = plt.subplots(figsize=(15, 5))

    if sample_stats:
        # Create a new column name including the number of publications 
        # that contain info (n) as well as the number of values available (m)
        to_plot.loc["new_col_names", :] = \
            to_plot.columns.values + \
            ", n: " + to_plot.loc["publications_number"].astype(int).astype(str) + \
            ", m: " + to_plot.loc["number_entries"].astype(int).astype(str)

        to_plot.columns = to_plot.loc["new_col_names"]
        to_plot = to_plot.drop(["number_entries", "publications_number", "new_col_names"], axis=0)

    if add_sources and "sources" in to_plot.columns:
        to_plot = to_plot.drop(["sources"], axis=0)

    if not swarmplot:
        _ = to_plot.plot(kind="box", ax=ax)
    else:
        if use_colors:
            _ = sns.boxplot(data=to_plot, ax=ax, width=0.5, boxprops=dict(alpha=0.2),
                            palette=sns.color_palette([el for el in colors.loc[
                                dict_info_years[year].columns, "Farbe (matplotlib strings)"].values]))
            _ = sns.swarmplot(data=to_plot, ax=ax,
                              palette=sns.color_palette([el for el in colors.loc[
                                  dict_info_years[year].columns, "Farbe (matplotlib strings)"].values]))
        else:
            _ = sns.boxplot(data=to_plot, ax=ax, width=0.5, boxprops=dict(alpha=0.2))
            _ = sns.swarmplot(data=to_plot, ax=ax)

    if show_title:
        if year == "SQ":
            _ = plt.title(title + " im Status quo")

        else:
            _ = plt.title(title + " im Jahr " + year)

    if use_limits:
        minimum = to_plot.min().min()
        maximum = to_plot.max().max()

        if minimum >= 0:
            ylim = [minimum - 0.1 * minimum,
                    maximum + 0.1 * maximum]
        else:
            ylim = [minimum + 0.1 * minimum,
                    maximum - 0.1 * maximum]

    _ = plt.ylim(ylim)
    _ = plt.xlabel("Lastmanagementkategorie")
    _ = plt.ylabel(ylabel)
    _ = plt.xticks(rotation=90)

    if savefig:
        plt.savefig(path_folder + filename + "_boxplot.png", dpi=150, bbox_inches="tight")

    plt.show()


def get_nlargest(dict_info_years, year="SQ", metric="50%", n=5, sample_stats=False):
    """
    Function takes the dictionary containing info on the relevant parameter
    indexed by years and detects the processes with the n largest potentials
    for the relevant parameter evaluated by a specific statistic metric
    (usually median or mean value).
     
    Parameters
    ----------   
    dict_info_years: dict or list of dicts
        Dictionary resp. dictionaries to store the info on the relevant 
        parameter, indexed by years
            
    year: str
        Year for which the processes with the largest potentials
        shall be detected
               
    metric: str
        Metric which shall be used for detecting the n largest processes
    
    n: int
        Determines how many values shall be used (n largest)
        
    sample_stats: boolean
        If True, ignore information on sample size included in the DataFrames
    
    Returns
    -------
    process_list: list
        List containing the processes with the largest potentials
    """

    to_describe = dict_info_years[year]

    # If sample_stats is True, drop rows holding publications number resp. number of entries
    if sample_stats:
        to_describe = to_describe.iloc[:-2]

    process_list = list(to_describe.describe().loc[metric].sort_values(
        ascending=False).index.values[:n])

    return process_list


def extract_stats(dict_info_years, years_dict):
    """
    Function takes parameter info as input and calculates info stats per year
    which are writen into a dictionary indexed by years and then merged together
    to obtain a DataFrame containing overall info.
    
    Parameters
    ----------    
    dict_info_years: dict
        Dictionary from which the information is obtained
    
    years_dict: dict
        Dictionary of years resp. year groups to be evaluated
    
    Returns
    -------
    dict stats_years: dict
        Dictionary containing the stats information of interest for every years
    
    df_stats_years pd.DataFrame
        pd.DataFrame containing the stats information on the parameter of 
        interest across all years
    """
    dict_stats_years = {}

    for year in years_dict:

        # Skip empty DataFrames since describe method cannot be applied to them
        if not dict_info_years[year].empty:
            # Obtain statistical info for each process using the describe method
            df = dict_info_years[year].describe().T[["min", "max", "mean", "50%"]].dropna(how="all")
            # Melt the DataFrame in order to prepare it for plotting the future development for the respective year (groups)
            dict_stats_years[year] = pd.melt(df.reset_index(), id_vars="Prozess", var_name="parameter").set_index(
                ["Prozess", "parameter"])

    df_stats_year = pd.DataFrame(columns=["Prozess", "parameter"])

    for k, v in dict_stats_years.items():
        df_stats_year = pd.merge(df_stats_year, v, on=["Prozess", "parameter"], suffixes=["", "_" + k], how="outer")

    df_stats_year = df_stats_year.set_index(["Prozess", "parameter"]).T

    # Change the index such that it includes years only
    df_stats_year = df_stats_year.reset_index()
    new_df = df_stats_year["index"].str.split("_", n=1, expand=True)
    new_df.at[0, 1] = "Status quo"
    df_stats_year["year_key"] = new_df[1]
    df_stats_year = df_stats_year.set_index("year_key", drop=True).drop("index", axis=1)

    return dict_stats_years, df_stats_year


def create_projection_boxplot(df_info, years_dict, process_cols, title, ylabel, colors,
                              ylim=[0, 3000], negate=False, use_colors=False, use_limits=True,
                              sample_stats=False, add_sources=False,
                              swarmplot=False, savefig=False, show_title=True,
                              path_folder="./out/",
                              filename="DR_parameter_projection"):
    """
    Function creates a boxplot for the projections on future development of
    a demand response parameter.
    
    Parameters
    ----------
    df_info: pd.DataFrame
        pd.DataFrame containing the information on the parameter of interest 
        across all publications (and for all years)
    
    years_dict: dict
        Dictionary of years resp. year groups to be evaluated
        
    process_cols: list
        The processes to be depicted in the plot. For the sake of readability,
        only a limited number of processes can be depicted at once (up to
        around five)
    
    title: str
        Title for the plot
    
    ylabel: str
        label for the yaxis

    colors: pd.DataFrame
        pd.DataFrame containing the category / color mapping
                
    ylim: list
        limits for the yaxis

    negate: boolean
        if True the values are negated, i.e. multiplied with -1 (for negative
        potential information)

    use_colors: boolean
        If True, colors from the given colors DataFrame will be used, default
        ones elsewhise
   
    use_limits: boolean
        If True, yaxis limits will be obtained from the extreme values of
        the data including some space
        
    sample_stats: boolean
        True if info on the sample size is included
    
    add_sources: boolean
        True if information on sources (page areas) is included
        
    swarmplot: boolean
        Create an overlay bee swarm plot if True using seaborn

    savefig: boolean
        Save figure to png if True
        
    show_title: boolean
        Show / hide the given title
        
    path_folder: str
        Path where the png file shall be stored
    
    filename: str
        Name of the png file to be stored    
    """

    # Manipulate the DataFrame in order to be able to plot it: select processes of interest
    to_plot = df_info.loc[[el for el in process_cols if el in df_info.index], :]

    # Workaround for sorting: introduce a temporary int column to sort by
    # and assign the status quo variable the lowest value to place it at the top
    to_plot["sort_col"] = to_plot["year_key"]
    to_plot.loc[to_plot["year_key"] == "SQ", "sort_col"] = 0
    to_plot["sort_col"] = to_plot["sort_col"].astype(int)

    to_plot = to_plot.set_index([to_plot.index, "year_key"]).sort_values(by=["sort_col", "Prozess"], ascending=True)
    to_plot = to_plot.drop("sort_col", axis=1)

    # transform the MultiIndex into a more readable form for the plot
    to_plot.index = to_plot.index.map(','.join)
    to_plot = to_plot.T

    if negate:
        rows_to_ignore = []

        if sample_stats:
            rows_to_ignore.extend(["publications_number", "number_entries"])

        # Workaround: to ignore publications number and number entries 
        # in negation, simply multiply with -1 twice
        to_plot = to_plot.mul(-1)
        to_plot.loc[rows_to_ignore] = to_plot.loc[rows_to_ignore].mul(-1)

    fig, ax = plt.subplots(figsize=(15, 5))

    if sample_stats:
        to_plot.loc["new_col_names", :] = \
            to_plot.columns.values + \
            ", n: " + to_plot.loc["publications_number"].astype(int).astype(str) + \
            ", m: " + to_plot.loc["number_entries"].astype(int).astype(str)

        to_plot.columns = to_plot.loc["new_col_names"]
        to_plot = to_plot.drop(["number_entries", "publications_number", "new_col_names"], axis=0)
        # Alternative: Simply drop the respective rows
        # to_plot = to_plot.iloc[:-2]

    if add_sources:
        to_plot = to_plot.iloc[:-1]

    if not swarmplot:
        _ = to_plot.plot(kind="box", ax=ax)
    else:
        if use_colors:
            _ = sns.boxplot(data=to_plot, ax=ax, width=0.5, boxprops=dict(alpha=0.2),
                            palette=sns.color_palette(
                                [el for el in colors.loc[sorted(process_cols)]["Farbe (matplotlib strings)"].values]))
            _ = sns.swarmplot(data=to_plot, ax=ax,
                              palette=sns.color_palette(
                                  [el for el in colors.loc[sorted(process_cols)]["Farbe (matplotlib strings)"].values]))
        else:
            _ = sns.boxplot(data=to_plot, ax=ax, width=0.5, boxprops=dict(alpha=0.2))
            _ = sns.swarmplot(data=to_plot, ax=ax)

    if use_limits:
        minimum = to_plot.min().min()
        maximum = to_plot.max().max()

        if minimum >= 0:
            ylim = [minimum - 0.1 * minimum,
                    maximum + 0.1 * maximum]
        else:
            ylim = [minimum + 0.1 * minimum,
                    maximum - 0.1 * maximum]

    if show_title:
        _ = plt.title(title)

    _ = plt.ylim(ylim)
    _ = plt.xlabel("Prozess")
    _ = plt.ylabel(ylabel)
    _ = plt.xticks(rotation=90)

    if savefig:
        plt.savefig(path_folder + filename + "_projection_box.png", dpi=150, bbox_inches="tight")

    plt.show()


def save_future_sources(df_info, process_cols, path_folder, filename,
                        add_sources=True):
    """
    Create and save sources information for the future outlook.
    """
    dict_sources = {}

    if add_sources:
        dict_sources["sources"] = df_info.loc[[el for el in process_cols if el in df_info.index], :][
            "sources"].unique().sum()

    save_sources(dict_sources, path_folder, filename)
