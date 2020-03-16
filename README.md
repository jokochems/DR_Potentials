# Demand response potential meta analysis

This repository contains all files from a **demand response meta-analysis
for Germany** in which 30 publications have been evaluated on demand response
parameter information.

## Contents and structure
The repository is structured as follows:
* main level: The main level contains input data as well as all routines for
extracting, manipulating, visualizing and storing demand response parameter
data.
* out: The folder out is the place where all results are stored. It contains
three subfolders: plots, stats as well as sources.

At the main level, the following files exist:
* "DR_potential_evaluation.ipynb": The main project file
* "Potenziale_Lastmanagement.xlsx": The raw data
* "Columns.xlsx": Information on the parameter column names used
* "potential_evaluation_funcs.py": User-defined functions which are imported
by the main project file.

## Usage
* Start the jupyter-notebook "DR_potential_evaluation.ipynb".
* Manipulate the parameter settings if necessary (i.e., the boolean parameters
controlling what is done).
* Run the notebook and obtain the data output demanded.

## Development
Feel free to manipulate the data according to your needs. If you want to
contribute, please create a fork, open a new branch and issue a pull request 
if you want to have your changes integrated. 

For further information on git
usage, please refer to the following sources:
* Quick and dirty main git commands: 
https://rogerdudler.github.io/git-guide/index.html
* Official git documentation: https://git-scm.com/docs
* Gitlab basic help: https://docs.gitlab.com/ee/gitlab-basics/README.html
* A very good and detailled book called ProGit: 
https://progit2.s3.amazonaws.com/en/2016-03-22-f3531/progit-en.1084.pdf