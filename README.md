# Probabilistic PV Disaggregation
This repository contains the code for the paper ["Probabilistic Disaggregation of Behind-the-Meter PV Systems Using Conformal Prediction"](https://doi.org/10.1109/TSG.2026.3676842).

# Structure of the project
- `notebooks/` - contains the main notebooks that conduct the experiments.
- `modules/` - contains the modules that are used in the experiments.
- `config/` - contains the column lists and customer lists used by the notebooks.
- `data/` - contains the Ausgrid data, Solcast irradiance/weather data merged with Ausgrid, and generated result files used in the experiments.
- `README.md` - this file, describing the project.


## Usage Instructions
1. Create a venv with `python -m venv env_name`.
2. Activate the venv with `source env_name/bin/activate`.
3. Install the dependencies with `pip install -r requirements.txt`.
4. Put the Ausgrid Solar Home Electricity Data [Ausgrid Solar Home Electricity Data](https://www.tandfonline.com/doi/abs/10.1080/14786451.2015.1100196) CSV files under the path `data/ausgrid/`. The notebooks expect the following raw files:
   1. `2010-2011 Solar home electricity data.csv`.
   2. `2011-2012 Solar home electricity data.csv`.
   3. `2012-2013 Solar home electricity data.csv`.
5. Put the Solcast irradiance/weather data [Solcast weather data](https://solcast.com/forecast-solar-irradiance-data) under the path `data/ausgrid/`. The merge notebook uses the weather columns listed in `config/weather_columns.txt` and expects the following files:
   1. `ausgrid_cluster5_meteo.csv`.
   2. `ausgrid_meteo_location2_2010_2012.csv`.
   3. `ausgrid_meteo_location3_2011_Jan.csv`.
6. Run the following Jupyter notebooks in the order below, to obtain the results.
   1. `ausgrid_load_2010_2011.ipynb`.
   2. `ausgrid_load_2011_2012.ipynb`.
   3. `ausgrid_load_2012_2013.ipynb`.
   4. `ausgrid_merge_3_years.ipynb`.
   5. `ausgrid_merge_elec_meteo.ipynb`.
   6. `ausgrid_preprocessing.ipynb`.
   7. `capacity_estimation_SA_aus.ipynb`.
   8. `ausgrid_point_regression.ipynb`.
   9. `ausgrid_prob_regression_lgb_two_parts12.ipynb`.
   10. `multiple_splits_aus_amb.ipynb`.
   11. `multiple_splits_aus_wis_bins.ipynb`.
The `ausgrid_exploration.ipynb` notebook is optional and can be used to inspect the processed Ausgrid data.


# Disclaimer about the data
The experiments in this repository use the Ausgrid Solar Home Electricity Data and irradiance/weather data from Solcast. If the raw Ausgrid data files or Solcast-derived merged file are not included in your local checkout, obtain them from the original data providers and place the processed files under `data/ausgrid/` before running the notebooks.

# Credit
We would like to thank the [data exploration repo](https://github.com/pierre-haessig/ausgrid-solar-data) by pierre-haessig for helping us getting started with the Ausgrid dataset faster.
