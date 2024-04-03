# Project Overview

This project is a collection of Python scripts used for processing and analyzing data from JPK force spectroscopy files. The scripts are designed to extract data from these files, perform various transformations and calculations, and generate plots for visualization.

## Main Files

- `mainScript.py`: the main script file

- `extractJPK.py`: file containing functions 
    - `force()`: 
    - (and hopefully soon also `QI()`)

- `plot.py`: file containing functions
    - `Fd(F, d, k, save='False')`: 
    - `Ft(F, t, k, save='False')`:
    - `Fdsubplot(F, d, k, F_sub)`:
    - `Ftsubplot(F, t, k, F_sub)`:

- `procBasic.py`: file containing functions
    - `max(F)`: 
    - `baselineSubtraction(F)`:
    - `smoothingSG`: 


## Libraries Used

- `matplotlib`: Used for creating plots for data visualization.
- `numpy`: Used for numerical operations.
- `pandas`: Used for data manipulation and analysis.
- `seaborn`: Used for statistical data visualization.
- `afmformats`: Used for loading data from JPK files.

## How to Run

To run these scripts, you will need to have Python installed along with the libraries mentioned in `requirements.txt`, preferably in a virtual environment.

Once the libraries are installed, you can run the scripts using Python:

Please note that you will need to have JPK files available in the specified directories for the scripts to work.