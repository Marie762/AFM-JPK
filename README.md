# Project Overview

This project is a collection of Python scripts used for processing and analyzing data from JPK force spectroscopy files. The scripts are designed to extract data from these files, perform various transformations and calculations, and generate plots for visualization.

## Main Files

- `mainScript.py`: the main script file

- `extractJPK.py`: file containing functions 
    - `force()`: returns `F, d, t`
    - `QI()`: returns `qmap`

- `metadata.py`: file containting functions
    - `JPKReaderList()`: returns `jpk_reader_list`
    - `Sensitivity()`: returns `sensitivity_list`
    - `SpringConstant()`: returns `spring_constant_list`
    - `Position()`: returns `position_list`
    - `Speed()`: returns `speed_list`
    - `Setpoint()`: returns `setpoint_list`

- `plot.py`: file containing functions
    - `Fd(F, d, save='False')`: returns `fig`
    - `Ft(F, t, save='False')`: returns `fig`
    - `Fdsubplot(F, d, F_sub, colour1='blue', colour2='orangered', colour3='indigo', subplot_name='subplot', save='False'))`: returns `fig`
    - `Ftsubplot(F, t, F_sub, colour1='blue', colour2='orangered', colour3='indigo', subplot_name='subplot', save='False')`: returns `fig`
    - `QIMap(data, ind, col, k, save='False')`: returns `fig`

- `procBasic.py`: file containing functions
    - `max(F)`: returns `max_value, max_element`
    - `baselineSubtraction(F)`: returns `F_bS`
    - `heightCorrection(d)`: returns `d_hC`
    - `heightZeroAtContactPoint(d, argmin_list)`: returns `d_hZ`
    - `tipDisplacement(F,d)`: returns `delta`
    - `smoothingSG(F, window_size, poly_order)`: returns `F_smoothSG`

- `contactPoint.py`: file containing functions
    - `baselineLinearFit(F, d, perc_bottom=0, perc_top=50, plot='False', saveplot='False')`: returns `M, B`
    - `contactPoint1(F, d, plot='False', saveplot='False', perc_bottom=0, perc_top=50)`: returns `argmin_list`
    - `contactPoint2(F, d, plot='False', saveplot='False')`: returns `argmin_list`
    - `QIcontactPoint1(F, d, perc_bottom=0, perc_top=50)`: returns `contact_point_height`
    - `QIcontactPoint2(F,d)`: returns `contact_point_height`

- `youngsModulus.py`: file containing functions
    - `func_power_law(x, a, b, c)`: returns `c +  a * (x**b)`
    - `func_parabolic(x, E, c0)`: returns `c0 + (4/3)*np.sqrt(R_c)*(E/(1-v**2))*x**(3/2)`
    - `func_conical(x, E, c0)`: returns `c0 + (E/(1-v**2))*((2*np.tan(alpha))/np.pi)*x**2`
    - `func_pyramid(x, E, c0)`: returns `c0 + ((np.tan(theta))/np.sqrt(2))*(E/(1-v**2))*x**2`
    - `fitYoungsModulus(F, delta, argmin_list)`: returns `popt_list, fig`
    - `variationYoungsModulus(F, delta, argmin_list, indenter='parabolic')`: returns `E, fig`

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