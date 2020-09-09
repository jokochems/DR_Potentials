# Routines for assessing demand response potentials

This repository contains routines for assessing demand response potentials.
These comprise:
* The outcome of a meta analysis on technical potentials and
* Routines for creating parameter sets using a clustering approach

The final result is a parameterization for demand response which can be
used to model it in the fundamental power market model _POMMES_.

## Demand response technical potential meta analysis

This repository contains all files from a **demand response meta-analysis
for Germany** in which 30 publications have been evaluated on demand response
parameter information (technical potentials).

## Demand response potential clustering

The results from the meta-analysis as well as some availability information
and assumptions are used to define demand response clusters, i.e. a grouping
of demand response categories, mostly at the level of applications resp.
processes.

## Contents and structure
The repository is structured as follows:
* main level: The main level contains input data as well as all routines for
extracting, manipulating, visualizing and storing demand response parameter
data.
* out: The folder out is the place where all results are stored. It contains
some subfolders: availability, plots, stats, parameterization as well as sources.
Availability input data for the clustering is already given as external input.
The other input data for the clustering is stored as well but can be easily
created from running the potential evaluation notebook as it is.

At the main level, the following files exist:
* "DR_potential_evaluation.ipynb": The notebook used for the potential 
meta-analysis
* "DR_potential_clustering.ipynb"; The notebook used for the potential 
clustering
* "Potenziale_Lastmanagement.xlsx": The raw data
* "Columns.xlsx": Information on the parameter column names used
* "potential_evaluation_funcs.py": User-defined functions which are imported
by the notebook for the meta-analysis
* "potential_clustering_funcs.py": User-defined functions which are imported
by the notebook for the potential clustering

## Usage
* Start the jupyter-notebook "DR_potential_evaluation.ipynb" or 
"DR_potential_clustering.ipynb"
* Manipulate the parameter settings if necessary (i.e., the boolean parameters
controlling what is done).
* Run the notebook and obtain the data output demanded.

## Development
Feel free to manipulate the data according to your needs. If you want to
contribute, please create a fork, open a new branch and issue a pull request 
if you want to have your changes integrated. Let me now, if you need other
permissions you do not yet have.

For further information on git
usage, please refer to the following sources:
* Quick and dirty main git commands: 
https://rogerdudler.github.io/git-guide/index.html
* Official git documentation: https://git-scm.com/docs
* Gitlab basic help: https://docs.gitlab.com/ee/gitlab-basics/README.html
* A very good and detailled book called ProGit: 
https://progit2.s3.amazonaws.com/en/2016-03-22-f3531/progit-en.1084.pdf