# Project Overview

This project is a collection of Python scripts used for processing and analyzing data from JPK force spectroscopy files. The scripts are designed to extract data from these files, perform various transformations and calculations, and generate plots for visualization.

## Main Files

- `extract data from jpk files.py`: This script is used to extract data from JPK force spectroscopy files. It uses the `afmformats` library to load the data and then processes it to extract the force and height measurements.

- `extract data from qi jpk files.py`: This script is similar to the previous one but is specifically designed to work with QI mode JPK files. It also uses the `afmformats` library to load the data and then processes it to extract the force and height measurements.

- `pffff.py`: This script is used for further processing and analysis of the data extracted from the JPK files. It includes functionality for finding the maximum force point, performing background subtraction, and applying a smoothing function.

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