import pandas as pd
import random
import os
from sklearn.metrics import r2_score
from crepes import WrapRegressor
from crepes.extras import margin, DifficultyEstimator, MondrianCategorizer
from mapie.regression import MapieQuantileRegressor
import lightgbm
from lightgbm import LGBMRegressor
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

my_confidence = 0.9
confidences = [i/100 for i in range(5, 100, 5)]
my_no_bins = 6
my_k = 12
my_seed = 42
# set the seed of numpy
np.random.seed(my_seed)


def adaptive_mondrian_bins(learner, X_cal, y_cal, seed, prob_method):
    list_bins = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 40, 60, 80, 100]
    # iterate over the list of bins, and return the bin with the lowest interval score
    min_wis = float('inf')
    optimal_no_bins = None
    for no_bins in list_bins:
        mc = MondrianCategorizer()
        mc.fit(X=X_cal, f=get_values, no_bins=no_bins)
        learner.calibrate(
            X_cal,
            y_cal,
            cps=True if prob_method == "cps_amb" else False,
            mc=mc,
            seed=seed,
        )
        wis = calc_multi_level_WIS(learner, X_cal, y_cal)
        if wis < min_wis:
            min_wis = wis
            optimal_no_bins = no_bins
    return optimal_no_bins


def cqr_multi_level_eval(X_test, y_test, confidences, X_prop_train, y_prop_train, X_cal, y_cal, calibrate=False, cap=1.1):
    qreg_sharpness, qreg_calibration_error, qreg_interval_score = calc_multi_level_cqr(
        X_test=X_test,
        y_test=y_test,
        confidences=confidences,
        X_prop_train=X_prop_train,
        y_prop_train=y_prop_train,
        X_cal=X_cal,
        y_cal=y_cal,
        cap=cap,
        calibrate=calibrate,
    )
    qreg_WIS = calc_multi_level_cqr_WIS(
        X_test=X_test,
        y_test=y_test,
        confidences=confidences,
        X_prop_train=X_prop_train,
        y_prop_train=y_prop_train,
        X_cal=X_cal,
        y_cal=y_cal,
        cap=cap,
        calibrate=calibrate,
    )
    return qreg_sharpness, qreg_calibration_error, qreg_interval_score, qreg_WIS


def cp_data_preparation(df_1, df_2,  list_of_features):
    df_aus_part1 = df_1
    df_aus_part2 = df_2
    df_aus_part1 = df_aus_part1[df_aus_part1["ghi"] >= 0.01]
    df_aus_part2 = df_aus_part2[df_aus_part2["ghi"] >= 0.01]
    df_aus_train = df_aus_part1[(df_aus_part1['datetime'] >= '2012-07-01')
                                & (df_aus_part1['datetime'] <= '2013-05-01')]
    # 2011-07-01 to 2012-07-01 06:00:00 from part2 as calibration
    df_aus_cal = df_aus_part1[(df_aus_part1['datetime'] >= '2013-05-01')
                              & (df_aus_part1['datetime'] <= '2013-07-01')]
    # 2018-12-23 to 2019-03-10 06:00:00 from part2 as test
    df_aus_test = df_aus_part2[(df_aus_part2['datetime'] >= '2013-05-01')
                               & (df_aus_part2['datetime'] <= '2013-07-01')]
    X_prop_train = df_aus_train[list_of_features]
    X_cal = df_aus_cal[list_of_features]
    X_test = df_aus_test[list_of_features]
    y_prop_train = df_aus_train["total_pv_gen_normalized"]
    y_cal = df_aus_cal["total_pv_gen_normalized"]
    y_test = df_aus_test["total_pv_gen_normalized"]
    len_test = len(df_aus_test)
    # convert y_test to numpy array
    y_test = np.array(y_test)
    return X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, len_test


def preprocess_two_groups(preprocessor_1, preprocessor_2, feature_columns):
    preprocessor_1.estimate_pv_capacity_from_pv_profiles_avg_high_values()
    preprocessor_2.estimate_pv_capacity_from_pv_profiles_avg_high_values()
    preprocessor_1.normalize_pv()
    preprocessor_2.normalize_pv()
    preprocessor_1.add_total_pv_gen()
    preprocessor_1.add_total_pv_gen_normalized(
        capacity=preprocessor_1.pv_capacity_from_pv_profiles)
    preprocessor_2.add_total_pv_gen()
    preprocessor_2.add_total_pv_gen_normalized(
        capacity=preprocessor_2.pv_capacity_from_pv_profiles)
    preprocessor_1.add_total_net()
    preprocessor_2.add_total_net()
    preprocessor_1.add_total_con()
    preprocessor_2.add_total_con()
    preprocessor_1.normalize_feature(
        feature_list=['total_net', 'total_pv_gen'])
    preprocessor_2.normalize_feature(
        feature_list=['total_net', 'total_pv_gen'])
    preprocessor_1.add_cos_sin_HoD()
    preprocessor_2.add_cos_sin_HoD()
    feature_columns_to_normalize = [
        col for col in feature_columns if col != 'total_net_norm']
    preprocessor_1.normalize_feature(feature_list=feature_columns_to_normalize)
    preprocessor_2.normalize_feature(feature_list=feature_columns_to_normalize)
    return preprocessor_1.data_pv_normalized, preprocessor_2.data_pv_normalized


def calc_percentage_error(real_capacities: np.ndarray, estimated_capacities: np.ndarray) -> float:
    """
    Calculates the percentage error between real and estimated capacities.

    This function computes the percentage error. If any of the input arrays contain NaN values,
    the function returns NaN.

    Parameters:
    real_capacities (np.ndarray or array-like): The actual capacities.
    estimated_capacities (np.ndarray or array-like): The estimated capacities.

    Returns:
    float: The percentage error, or NaN if any input contains NaN values.
    """
    if np.isnan(real_capacities).any() or np.isnan(estimated_capacities).any():
        return np.nan
    return (real_capacities.sum() - estimated_capacities.sum()) / real_capacities.sum() * 100


def calc_mean_percentage_error(real_capacities: np.ndarray, estimated_capacities: np.ndarray) -> float:
    """
    Calculates the mean percentage error between real and estimated capacities.

    This function computes the mean percentage error averaging over the individual
    percentage errors for each household in the input arrays.
    If any of the input arrays contain NaN values, the function returns NaN.

    Parameters:
    real_capacities (array-like): The actual capacities.
    estimated_capacities (array-like): The estimated capacities.

    Returns:
    float: The mean percentage error, or NaN if any input contains NaN values.
    """
    if np.isnan(real_capacities).any() or np.isnan(estimated_capacities).any():
        return np.nan
    # n is the length of real_capacities
    n = real_capacities.size
    # individual_errors is the percentage error of each household
    individual_errors = (real_capacities - estimated_capacities) / \
        real_capacities  # Avoid division by zero
    return individual_errors.sum()/n*100


def calc_error_factor(real_capacities_sum: float, estimated_capacities_sum: float) -> float:
    """
    Calculates the error factor between real and estimated capacities.

    This function computes the error factor by dividing the sum of real capacities
    by the sum of estimated capacities.

    Parameters:
    real_capacities_sum (float): The sum of real capacities.
    estimated_capacities_sum (float): The sum of estimated capacities.

    Returns:
    float: The error factor, or NaN if any input contains NaN values.
    """
    return real_capacities_sum/estimated_capacities_sum


def calc_r2_score(real_capacities: np.ndarray, estimated_capacities: np.ndarray) -> float:
    """
    Calculates the R2 score between real and estimated capacities.

    This function computes the R2 score by comparing the sum of real capacities
    to the sum of estimated capacities.

    Parameters:
    real_capacities (np.ndarray or array-like): The actual capacities.
    estimated_capacities (np.ndarray or array-like): The estimated capacities.

    Returns:
    float: The R2 score, or NaN if any input contains NaN values.
    """
    # if there is any NaN in real_capacities or estimated_capacities, return NaN
    # otherwise, return the R2 score
    if np.isnan(real_capacities).any() or np.isnan(estimated_capacities).any():
        return np.nan
    return r2_score(real_capacities, estimated_capacities)


def add_trendline(x: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
    """
    Adds a quadratic trendline to a given plot.

    This function sorts the input data, fits a second-order polynomial
    to the sorted data, and plots the resulting trendline on the provided
    Axes object.

    Parameters:
    x (np.ndarray or array-like): The x-coordinates of the data points.
    y (np.ndarray or array-like): The y-coordinates of the data points.
    ax (matplotlib.axes.Axes): The Axes object on which to plot the trendline.

    Returns:
    None
    """
    # Sort x and y values to ensure a continuous line
    # convert x and y to numpy array
    x = np.array(x)
    y = np.array(y)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Fit polynomial and create trendline
    z = np.polyfit(x_sorted, y_sorted, 2)
    p = np.poly1d(z)
    ax.plot(x_sorted, p(x_sorted), "--", color='red', linewidth=2)


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, fig_name: str = '../figs/actual_vs_predicted.pdf', xlabel: str = 'Actual', ylabel: str = 'Estimated', trendline: bool = True) -> None:
    """
    Plots the actual vs. predicted values and optionally a trendline.
    This function creates a scatter plot of the actual vs. predicted values,
    draws a reference line of y=x, and optionally adds a quadratic trendline.
    The plot is saved as a PDF file.

    Parameters:
    y_test (array-like): The actual values.
    y_pred (array-like): The predicted values.
    fig_name (str): The file path where the plot will be saved. Default is '../figs/actual_vs_predicted.pdf'.
    xlabel (str): The label for the x-axis. Default is 'Actual'.
    ylabel (str): The label for the y-axis. Default is 'Estimated'.
    trendline (bool): Whether to add a trendline to the plot. Default is True.

    Returns:
    None
    """
    # plot the actual vs predicted, with a line of y=x
    plt.style.use(['science'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [
        y_test.min(), y_test.max()], 'k-', lw=2)
    if trendline:
        add_trendline(y_test, y_pred, ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(fig_name, format='pdf')
    plt.show()


def read_column_list_from_config(filename):
    """Read column list from config file"""
    config_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'config')
    filepath = os.path.join(config_dir, filename)

    with open(filepath, 'r') as f:
        columns = [line.strip() for line in f.readlines() if line.strip()]
    return columns


def add_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:

    # add HoD dow doy month	year, and move the columns to the front
    # use the datetime column to add the columns
    df['HoD'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['doy'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    # move the columns to the front
    cols_to_move = ['HoD', 'dow', 'doy', 'month', 'year']
    cols_to_keep = ['datetime']
    df = df[cols_to_keep + cols_to_move +
            [col for col in df.columns if col not in cols_to_move + cols_to_keep]]
    return df


def df_common_xylabel_plot(df, y, doy_start=0, doy_end=366, year=[2017, 2018, 2019], xlabel='Datetime', ylabel="example ylabel", layout=(7, 4), subplots=True, figsize=(20, 20)):
    # make a list to plot, excluding the columns of datetime and Timestamp
    axes = df[(df['doy'] >= doy_start) & (df['doy'] <= doy_end) & (df['year'].isin(year))].plot(
        x='datetime', y=y, subplots=subplots, figsize=figsize, layout=layout, sharex=True, xlabel='')
    if subplots == True:
        fig = axes[0, 0].get_figure()
    else:
        fig = axes.get_figure()  # to avoid the problem of "'Axes' object is not subscriptable"
    # add a common y label
    fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', size=20)
    # add a common x label
    fig.text(0.5, 0.0, xlabel, ha='center', size=20)


def split_households(df: pd.DataFrame, n_1, n_2, seed, date_columns: list[str], weather_columns: list[str]) -> list[pd.DataFrame]:
    # n_1: number of households in the first part
    # n_2: number of households in the second part
    # Keep the weather data and the datetime column for both parts
    # remove the extra columns first
    df_extra_columns = df[weather_columns+date_columns]
    df_1_without_extra_columns = df.drop(columns=weather_columns+date_columns)
    df_2_without_extra_columns = df.drop(columns=weather_columns+date_columns)

    # Set random seed for reproducibility
    random.seed(seed)

    # Extract unique household IDs from PV columns (assuming they follow pattern like "household_id_pv")
    pv_cols = [col for col in df.columns if col.endswith('_pv')]
    # Extract household IDs by removing the '_pv' suffix
    household_ids = [col.replace('_pv', '') for col in pv_cols]

    # Randomly sample household IDs ensuring no overlap
    households_1 = random.sample(household_ids, n_1)
    # Sample households_2 from remaining households to avoid overlap
    remaining_households = [h for h in household_ids if h not in households_1]
    households_2 = random.sample(remaining_households, n_2)

    # Create column lists for each household type using the same household IDs
    pv_cols_1 = [household_id + '_pv' for household_id in households_1]
    con_cols_1 = [household_id + '_con' for household_id in households_1]
    net_cols_1 = [household_id + '_net' for household_id in households_1]

    pv_cols_2 = [household_id + '_pv' for household_id in households_2]
    con_cols_2 = [household_id + '_con' for household_id in households_2]
    net_cols_2 = [household_id + '_net' for household_id in households_2]

    df_1 = df_1_without_extra_columns[pv_cols_1 + con_cols_1 + net_cols_1]
    df_2 = df_2_without_extra_columns[pv_cols_2 + con_cols_2 + net_cols_2]

    # add the extra columns back to df_1 and df_2
    df_1 = pd.concat([df_1, df_extra_columns], axis=1)
    df_2 = pd.concat([df_2, df_extra_columns], axis=1)
    return df_1, df_2


# Section: Functions for Mondrian Categorizer


def get_values(X):
    # The function get_values(X) is returning X[:, 0], which takes only the first column of X. Ensure that:
    # X is indeed a NumPy array or some data structure that supports indexing with [:,0].
    # If X is not a NumPy array, this operation could raise an error.
    # convert X to numpy array
    X = np.array(X)
    # returns only the first column of X
    # where the first column represents a significant feature derived from the Laplace matrix.
    # Leading eigenvalue from Laplace matrix
    return X[:, 0]


# Section:Implementation of probabilistic metrics

def calc_penalty(y_test, lower, upper, confidence_i):
    # if the prediction is outside the interval, calculate the penalty
    y_test = np.asarray(y_test).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    penalty = 0.0
    alpha = 1 - confidence_i
    n = len(y_test)
    if n == 0:
        return np.nan
    for i in range(n):
        if y_test[i] < lower[i]:
            penalty += 2*(lower[i]-y_test[i])/alpha
        elif y_test[i] > upper[i]:
            penalty += 2*(y_test[i]-upper[i])/alpha
    return penalty/len(y_test)


def calculate_coverage(y_test, intervals):
    # calculate the coverage
    y_test = np.asarray(y_test).ravel()
    return np.sum([1 if (y_test[i] >= intervals[i, 0] and
                         y_test[i] <= intervals[i, 1]) else 0
                   for i in range(len(y_test))])/len(y_test)


def calculate_mean_size(intervals):
    return (intervals[:, 1]-intervals[:, 0]).mean()


def calculate_median_size(intervals):
    return np.median((intervals[:, 1]-intervals[:, 0]))

# Section: Functions for multi-level evaluation


def calc_multi_level(learner, X_test, y_test, cap=1.1):
    coverages = []
    mean_sizes = []
    penalties = []
    for confidence_i in confidences:
        cp_int = learner.predict_int(
            X_test, y_min=0, y_max=cap, confidence=confidence_i, seed=my_seed)
        lower = cp_int[:, 0]
        upper = cp_int[:, 1]
        coverages.append(calculate_coverage(y_test, cp_int))
        mean_sizes.append(calculate_mean_size(cp_int))
        penalties.append(calc_penalty(y_test, lower, upper, confidence_i))
    sharpness = np.mean(mean_sizes)
    calibration_error = np.mean(penalties)
    interval_score = sharpness+calibration_error
    return sharpness, calibration_error, interval_score


def calc_multi_level_WIS(learner, X_test, y_test, cap=1.1):
    K = len(confidences)
    interval_scores = []
    wk_interval_scores = np.zeros((len(y_test), len(confidences)))
    w0 = 1/2
    for idx, confidence_i in enumerate(confidences):
        cp_int = learner.predict_int(
            X_test, y_min=0, y_max=cap, confidence=confidence_i, seed=my_seed)
        lower = cp_int[:, 0]
        upper = cp_int[:, 1]
        width = upper-lower
        alpha_i = 1-confidence_i
        # vectorized penalty calculation
        penalties = np.zeros_like(y_test)
        mask_lower = y_test < lower
        mask_upper = y_test > upper
        penalties[mask_lower] = 2 * \
            (lower[mask_lower] - y_test[mask_lower])/alpha_i
        penalties[mask_upper] = 2 * \
            (y_test[mask_upper] - upper[mask_upper])/alpha_i
        interval_score = width + penalties
        interval_scores.append(interval_score)
        wk_interval_scores[:, idx] = alpha_i/2 * interval_score

    interval_scores = np.array(interval_scores)
    # m is the predictive median
    # for all confidences, take the median of the prediction intervals
    # median in confidence dimension
    m = learner.predict(X_test)
    # vectorized weighted interval score calculation
    weighted_interval_scores = 1 / \
        (K+1/2) * (w0*np.abs(y_test-m) + np.sum(wk_interval_scores, axis=1))

    return np.mean(weighted_interval_scores)


def calc_multi_level_cqr(X_test, y_test, confidences, X_prop_train, y_prop_train, X_cal, y_cal, calibrate=True, cap=1.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use numpy arrays so LGBM is fitted without feature names and no sklearn warning at predict
    X_prop_train = np.asarray(X_prop_train, dtype=float)
    X_cal = np.asarray(X_cal, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_prop_train = np.asarray(y_prop_train).ravel()
    y_cal = np.asarray(y_cal).ravel()
    y_test = np.asarray(y_test).ravel()
    mask_prop = np.isfinite(y_prop_train)
    X_prop_train = X_prop_train[mask_prop]
    y_prop_train = y_prop_train[mask_prop]
    mask_cal = np.isfinite(y_cal)
    X_cal = X_cal[mask_cal]
    y_cal = y_cal[mask_cal]
    coverages = []
    mean_sizes = []
    penalties = []
    for confidence_i in confidences:
        alpha = 1-confidence_i
        # if calibrate is True, use X_cal and y_cal for calibration
        estimator = lightgbm.LGBMRegressor(
            objective='quantile', alpha=alpha, random_state=my_seed, verbose=-1)
        if calibrate:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train,
                        X_calib=X_cal, y_calib=y_cal, random_state=my_seed)
            cqr_pred, cqr_int = cqr_reg.predict(X_test, alpha=alpha)
        else:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train, random_state=my_seed)
            # when not calibrated with calibration set, do not specify alpha here.
            cqr_pred, cqr_int = cqr_reg.predict(X_test)
        # limit the prediction interval to the range [0,cap]
        cqr_int = np.clip(cqr_int, 0, cap)
        lower = cqr_int[:, 0]
        upper = cqr_int[:, 1]
        coverages.append(calculate_coverage(y_test, cqr_int))
        mean_sizes.append(calculate_mean_size(cqr_int))
        penalties.append(calc_penalty(y_test, lower, upper, confidence_i))
    sharpness = np.mean(mean_sizes)
    calibration_error = np.mean(penalties)
    interval_score = sharpness+calibration_error
    return sharpness, calibration_error, interval_score


def calc_multi_level_cqr_WIS(X_test, y_test, confidences, X_prop_train, y_prop_train, X_cal, y_cal, calibrate=True, cap=1.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use numpy arrays so LGBM is fitted without feature names and no sklearn warning at predict
    X_prop_train = np.asarray(X_prop_train, dtype=float)
    X_cal = np.asarray(X_cal, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_prop_train = np.asarray(y_prop_train).ravel()
    y_cal = np.asarray(y_cal).ravel()
    y_test = np.asarray(y_test).ravel()
    mask_prop = np.isfinite(y_prop_train)
    X_prop_train = X_prop_train[mask_prop]
    y_prop_train = y_prop_train[mask_prop]
    mask_cal = np.isfinite(y_cal)
    X_cal = X_cal[mask_cal]
    y_cal = y_cal[mask_cal]
    K = len(confidences)
    interval_scores = []
    wk_interval_scores = np.zeros((len(y_test), len(confidences)))
    w0 = 1/2
    for idx, confidence_i in enumerate(confidences):
        alpha = 1-confidence_i
        # if calibrate is True, use X_cal and y_cal for calibration
        estimator = lightgbm.LGBMRegressor(
            objective='quantile', alpha=alpha, random_state=my_seed, verbose=-1)
        if calibrate:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train,
                        X_calib=X_cal, y_calib=y_cal, random_state=my_seed)
            cqr_pred, cqr_int = cqr_reg.predict(X_test, alpha=alpha)
        else:
            cqr_reg = MapieQuantileRegressor(
                estimator=estimator, alpha=alpha)
            cqr_reg.fit(X=X_prop_train, y=y_prop_train, random_state=my_seed)
            cqr_pred, cqr_int = cqr_reg.predict(X_test)
        # limit the prediction interval to the range [0,1]
        cqr_int = np.clip(cqr_int, 0, cap)
        lower = cqr_int[:, 0]
        upper = cqr_int[:, 1]
        # reshape to match the shape of y_test. e.g. from (757, 1) to (757,)
        lower = lower.reshape(-1)
        upper = upper.reshape(-1)
        width = upper-lower
        alpha_i = 1-confidence_i
        # vectorized penalty calculation
        penalties = np.zeros_like(y_test)
        mask_lower = y_test < lower
        mask_upper = y_test > upper
        penalties[mask_lower] = 2 * \
            (lower[mask_lower] - y_test[mask_lower])/alpha_i
        penalties[mask_upper] = 2 * \
            (y_test[mask_upper] - upper[mask_upper])/alpha_i
        interval_score = width + penalties
        interval_scores.append(interval_score)
        wk_interval_scores[:, idx] = alpha_i/2 * interval_score
    interval_scores = np.array(interval_scores)
    # m is the predictive median
    # for all confidences, take the median of the prediction intervals
    # median in confidence dimension
    m = cqr_pred
    # vectorized weighted interval score calculation
    weighted_interval_scores = 1 / \
        (K+1/2) * (w0*np.abs(y_test-m) + np.sum(wk_interval_scores, axis=1))

    return np.mean(weighted_interval_scores)


def calc_size_stratified_coverage(intervals, y_test):
    # divide y_test and intervals into 3 even parts
    # with each bin, calculate the coverage
    ssc = []
    len_test = len(y_test)
    for i in range(3):
        # get the ith 1/3 of y_test
        y_test_part = y_test[i * len_test // 3: (i + 1) * len_test // 3]
        # get the ith 1/3 of intervals
        intervals_part = intervals[i * len_test // 3: (i + 1) * len_test // 3]
        # calculate the coverage
        ssc.append(calculate_coverage(y_test_part, intervals_part))
    # extract the lowest coverage and return it
    lowest_coverage = min(ssc)
    return lowest_coverage
