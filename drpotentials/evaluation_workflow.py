"""
Workflow wrappers for the evaluation of technical demand response potentials.

Note:
    - For a dedicated description of the analysis, see jupyter notebook.
    - This script makes use of individual analysis steps, defined in a
      dedicated function module.

Notebook: DR_potential_evaluation.ipynb

@author: Johannes Kochems
"""
import pandas as pd

from drpotentials.evaluation_funcs import (
    extract_info,
    add_value_counts_and_sources,
    groupby_process_category,
    transpose_and_split,
    extract_sample_stats,
    extract_sample_sources,
    create_boxplot,
    get_nlargest,
)


def run_analyses_for_parameter_single_year(
    df,
    swapped_cols_dict,
    parameters,
    sector,
    years_dict,
    year,
    plot_title,
    plot_ylabel,
    plot_colors,
    filter_sector=None,
    drop_data_lack=True,
    save_stats=True,
    path_folder_stats="./out/stats/",
    file_name_stats="stats.csv",
    save_sources=True,
    path_folder_sources="./out/sources/",
    file_name_sources="sources.txt",
    show_plot=True,
    ylim=[0, 3000],
    use_colors=True,
    use_limits=True,
    swarmplot=True,
    savefig=False,
    show_title=True,
    path_folder_plots="./out/plots/",
    file_name_plot="parameter",
    return_dfs=False,
    use_category_shortcuts_for_plots=True,
    negate=False,
    format_yaxis=False,
    format_yaxis_decimals=False,
):
    """Wrapper function that runs an analyses for a certain parameter

    Analysis is limited to the year given

    Complete analysis chain consists of:
    * Extracting the parameter data
    * Adding sources information to it
    * Grouping parameter data by process category
    * Transposing data and splitting numeric data, counts and sources
    * Calculating sample stats
    * Compiling sample sources
    * Creating a nice combined swarm and boxplot for visualization

    See functions
    * exctract_info
    * add_value_counts
    * groupby_process_category
    * transpose_and_split
    * extract_sample_stats
    * extract_sample_sources
    * create_boxplot
    and their parameters

    Returns
    -------
    Nothing by default. If return_dfs is True, returns the following

    numeric_df : pd.DataFrame
        All numeric parameter data

    counts_df : pd.DataFrame
        Number of publications and data points

    sample_df : pd.DataFrame
        statistical moments of numeric parameter data

    sample_sources : str
        A combined string with the parameter sources information
    """
    filtered_df = extract_info(
        df,
        swapped_cols_dict,
        parameters,
        sector,
        years_dict,
        year,
        filter_sector,
    )
    filtered_df = add_value_counts_and_sources(filtered_df, drop_data_lack)
    grouped_df = groupby_process_category(filtered_df)
    numeric_df, counts_df, sources_df = transpose_and_split(grouped_df)

    sample_df = extract_sample_stats(
        numeric_df,
        save=save_stats,
        path_folder=path_folder_stats,
        file_name=file_name_stats,
    )

    sample_sources = extract_sample_sources(
        sources_df,
        save=save_sources,
        path_folder=path_folder_sources,
        file_name=file_name_sources,
    )

    if use_category_shortcuts_for_plots:
        numeric_df, counts_df, plot_colors = rename_using_shortcuts(numeric_df, counts_df, plot_colors)

    if show_plot:
        create_boxplot(
            numeric_df,
            counts_df,
            plot_title,
            plot_ylabel,
            colors=plot_colors,
            year=year,
            ylim=ylim,
            use_colors=use_colors,
            use_limits=use_limits,
            swarmplot=swarmplot,
            savefig=savefig,
            show_title=show_title,
            negate=negate,
            format_yaxis=format_yaxis,
            format_yaxis_decimals=format_yaxis_decimals,
            path_folder=path_folder_plots,
            file_name=file_name_plot,
        )

    if return_dfs:
        return numeric_df, counts_df, sample_df, sample_sources


def rename_using_shortcuts(
    numeric_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    plot_colors: pd.DataFrame,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Rename columns in data sets used to create box and swarmplots"""
    numeric_df = numeric_df.rename(
        columns=plot_colors["Prozesskategorie short"].to_dict(),
    )
    counts_df = counts_df.rename(
        columns=plot_colors["Prozesskategorie short"].to_dict(),
    )
    plot_colors = plot_colors.rename(columns={"Prozesskategorie short": "Prozesskategorie"}).set_index(
        "Prozesskategorie", drop=True
    )
    return numeric_df, counts_df, plot_colors


def extract_data_for_parameter_all_years(
    df,
    swapped_cols_dict,
    parameters,
    sector,
    years_dict,
    drop_data_lack=True,
    file_name_stats="stats",
    file_name_sources="sources",
    filter_sector=None,
):
    """Wrapper function that extracts data for certain parameter and all years

    Analysis chain consists of:
    * Extracting the parameter data
    * Adding sources information to it
    * Grouping parameter data by process category
    * Transposing data and splitting numeric data, counts and sources
    * Calculating sample stats
    * Compiling sample sources

    See functions
    * exctract_info
    * add_value_counts
    * groupby_process_category
    * transpose_and_split
    * extract_sample_stats
    * extract_sample_sources
    and their parameters
    """
    for year in years_dict.keys():
        filtered_df = extract_info(
            df,
            swapped_cols_dict,
            parameters,
            sector,
            years_dict,
            year,
            filter_sector,
        )
        filtered_df = add_value_counts_and_sources(filtered_df, drop_data_lack)
        grouped_df = groupby_process_category(filtered_df)
        numeric_df, counts_df, sources_df = transpose_and_split(grouped_df)

        extract_sample_stats(
            numeric_df,
            save=True,
            file_name=file_name_stats + "_" + year + ".csv",
        )

        extract_sample_sources(
            sources_df,
            save=True,
            file_name=file_name_sources + "_" + year + ".txt",
        )


def extract_projection_for_all_years(
    df,
    swapped_cols_dict,
    parameters,
    sector,
    years_dict,
    plot_title,
    plot_ylabel,
    plot_colors,
    processes,
    filter_sector=None,
    drop_data_lack=True,
    save_sources=True,
    path_folder_sources="./out/sources/",
    file_name_sources="sources_projection.txt",
    ylim=[0, 3000],
    use_colors=True,
    use_limits=True,
    swarmplot=True,
    savefig=False,
    show_title=True,
    path_folder_plots="./out/plots/",
    file_name_plot="parameter_projection",
    return_data=False,
    use_category_shortcuts_for_plots=True,
    negate=False,
    format_yaxis=False,
):
    """Extract data on certain pamareter(s) and processes for all years

    Run entire analysis chain including plotting projections

    Complete analysis chain consists of:
    * Extracting the parameter data
    * Adding sources information to it
    * Grouping parameter data by process category
    * Transposing data and splitting numeric data, counts and sources
    * Calculating sample stats
    * Compiling sample sources
    * Creating a nice combined swarm and boxplot for visualization

    See functions
    * extract_info
    * add_value_counts
    * groupby_process_category
    * transpose_and_split
    * extract_sample_stats
    * extract_sample_sources
    * create_boxplot
    and their parameters

    Additional parameters
    ---------------------
    processes: list of str
        Processes for which the projection shall be evaluated
    """
    # Filter for processes
    df = df.loc[df.index.isin(processes)]

    # Use combined data sets and add entries for years
    combined_numeric_df = pd.DataFrame()
    combined_counts_df = pd.DataFrame()
    color_palette = pd.DataFrame(columns=plot_colors.columns)

    for year in years_dict.keys():
        filtered_df = extract_info(
            df,
            swapped_cols_dict,
            parameters,
            sector,
            years_dict,
            year,
            filter_sector,
        )
        filtered_df = add_value_counts_and_sources(filtered_df, drop_data_lack)
        grouped_df = groupby_process_category(filtered_df)
        numeric_df, counts_df, sources_df = transpose_and_split(grouped_df)

        sample_sources = extract_sample_sources(sources_df, save=False, return_sources=True)

        if use_category_shortcuts_for_plots:
            (
                numeric_df,
                counts_df,
                plot_colors_year,
            ) = rename_using_shortcuts(numeric_df, counts_df, plot_colors)
        else:
            plot_colors_year = plot_colors.copy()

        # Add year information to data
        numeric_df.columns = numeric_df.columns + " - " + year
        counts_df.columns = counts_df.columns + " - " + year

        combined_numeric_df = pd.concat([combined_numeric_df, numeric_df], axis=1)
        combined_counts_df = pd.concat([combined_counts_df, counts_df], axis=1)

        if year != "SQ":
            if sample_sources is not None:
                combined_sources_string = combined_sources_string + "; " + sample_sources
        else:
            if sample_sources is not None:
                combined_sources_string = sample_sources
            else:
                combined_sources_string = ""

        # Prepare colors for boxplot
        plot_colors_year.index = plot_colors_year.index + " - " + year
        color_palette = pd.concat([color_palette, plot_colors_year])

    # Remove duplicates from (already tidied up) sources information
    unique_sources_set = set(combined_sources_string.split("; "))
    unique_sources_string = "; ".join(unique_sources_set)

    create_boxplot(
        combined_numeric_df,
        combined_counts_df,
        plot_title,
        plot_ylabel,
        colors=color_palette,
        ylim=ylim,
        use_colors=use_colors,
        use_limits=use_limits,
        swarmplot=swarmplot,
        savefig=savefig,
        show_title=show_title,
        include_year=False,
        negate=negate,
        format_yaxis=format_yaxis,
        path_folder=path_folder_plots,
        file_name=file_name_plot,
    )

    if save_sources:
        with open(path_folder_sources + file_name_sources, "w", encoding="UTF8") as opf:
            opf.write(unique_sources_string)

    if return_data:
        return combined_numeric_df, combined_counts_df, unique_sources_string


def extract_nlargest(
    df,
    sector,
    years_dict,
    swapped_cols_dict,
    filter_sector=None,
    drop_data_lack=True,
    metric="50%",
    n=5,
):
    """Get the process categories with the largest potentials

    Parameters
    ----------
    df : pandas.DataFrame
        Grouped input data

    sector: str
        The sector for which to extract the processes
        with the largest potentials

    years_dict : dict
        Dictionary mapping years to a string

    swapped_cols_dict : dict
        Dictionary mapping columns to a parameter group

    filter_sector : dict
        Dictionary for sectors

    drop_data_lack : boolean
        If True, drop processes with less than three data points

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
    for year in ["SQ", "2050"]:
        filtered_df = extract_info(
            df,
            swapped_cols_dict,
            [
                "Potenzial positiv Mittel",
                "Potenzial positiv min",
                "Potenzial positiv max",
            ],
            sector,
            years_dict,
            year,
            filter_sector,
        )
        filtered_df = add_value_counts_and_sources(filtered_df, drop_data_lack)
        grouped_df = groupby_process_category(filtered_df)
        numeric_df, counts_df, sources_df = transpose_and_split(grouped_df)

        stats_df = extract_sample_stats(
            numeric_df,
            save=False,
        )

        process_list.extend(get_nlargest(stats_df, metric, n))

    # Remove duplicates
    process_list = list(set(process_list))

    return process_list
