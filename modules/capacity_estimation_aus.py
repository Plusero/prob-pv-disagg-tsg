import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils import calc_percentage_error, calc_mean_percentage_error,  calc_r2_score, calc_error_factor


def capacity_estimation_error_factor(df1, df2):
    df_part1 = df1
    df_part2 = df2
    f_name_pv_capacity = '../data/ausgrid/ausgrid_pv_capacity_from_pv_profiles.csv'
    df_pv_capacity = pd.read_csv(f_name_pv_capacity)
    df_pv_capacity.rename(columns={
        df_pv_capacity.columns[0]: 'household', df_pv_capacity.columns[1]: 'capacity'}, inplace=True)

    households_part2 = set([int(col.replace('_net', ''))
                            for col in df_part2.columns if col.endswith('_net') and col != 'total_net'])
    df_pv_capacity_part2 = df_pv_capacity[df_pv_capacity['household'].isin(
        households_part2)]
    #####################
    households_to_remove = []
    households_to_remove_with_suffix = [
        col + '_net' for col in households_to_remove]
    # columns that ends with '_net'
    net_cols = [col for col in df_part1.columns if col.endswith('_net')]
    # remove total_net and households to remove from net_cols
    net_cols_part1 = [col for col in net_cols if col !=
                      'total_net' and col not in households_to_remove_with_suffix]
    net_cols_without_total_net_part1 = [
        col for col in net_cols_part1 if col != 'total_net']
    cap_est_based_load_part1_tmp = capacity_estimation_base_load(
        df_part1, net_cols_without_total_net_part1)
    cap_est_based_load_part1_tmp.estimate_capacity()
    base_load_correction_factor_from_part1 = cap_est_based_load_part1_tmp.base_load_correction_factor
    ####################
    households_to_remove_with_suffix = [
        col + '_net' for col in households_to_remove]
    # columns that ends with '_net'
    net_cols = [col for col in df_part2.columns if col.endswith('_net')]
    # remove total_net and households to remove from net_cols
    net_cols_part2 = [col for col in net_cols if col !=
                      'total_net' and col not in households_to_remove_with_suffix]
    net_cols_without_total_net = [
        col for col in net_cols_part2 if col != 'total_net']
    cap_est_based_load_part2 = capacity_estimation_base_load(
        df_part2, net_cols_without_total_net, base_load_correction_factor=base_load_correction_factor_from_part1)
    capacity_based_load_part2 = cap_est_based_load_part2.estimate_capacity()
    household_ids_from_capacity = [
        int(col.replace('_net', '')) for col in capacity_based_load_part2.index]
    df_capacity_based_load_part2 = pd.DataFrame({
        'household': household_ids_from_capacity,
        'estimated_capacity': capacity_based_load_part2.values
    })

    # Merge with df_pv_capacity_part2 based on household ID
    df_pv_capacity_part2 = df_pv_capacity_part2.merge(
        df_capacity_based_load_part2, on='household', how='left')
    df_pv_capacity_part2['estimated_capacity'] = df_pv_capacity_part2['estimated_capacity'].clip(
        lower=0)
    # Order df_pv_capacity_part2 to match the order of capacity_based_load_part2
    household_ids_from_capacity = [
        int(col.replace('_net', '')) for col in capacity_based_load_part2.index]
    df_pv_capacity_part2 = df_pv_capacity_part2.set_index(
        'household').reindex(household_ids_from_capacity).reset_index()
    real_capacity_part2 = df_pv_capacity_part2['capacity'].to_numpy()
    estimated_capacity_part2 = df_pv_capacity_part2['estimated_capacity'].to_numpy(
    )
    real_capacity_part2_sum = df_pv_capacity_part2['capacity'].sum()
    estimated_capacity_part2_sum = df_pv_capacity_part2['estimated_capacity'].sum(
    )
    error_factor_part2 = calc_error_factor(
        real_capacity_part2_sum, estimated_capacity_part2_sum)
    return error_factor_part2


class capacity_estimation_base_load:
    """
    A class for estimating the capacity of PV systems.
    Step 1: Sample the net load at night time as base load, when ghi is low and PV gen is (nearly) zero.
    Step 2: Sample the load when the ghi is below a certain threshold as noon load.
    Step 3: Calculate the capacity, which is (base load - noon load) * correction factor,
    where the correction factor is (max_irradiance/sample_irradiance). This correction factor originates the (nearly) linear relationship between ghi and PV gen.
    """

    def __init__(self, df: pd.DataFrame, cols: list, irradiance_threshold_noon: float = 500, irradiance_threshold_night: float = 0.01, max_irradiance: float = 1000, base_load_correction_factor: float = None):
        self.df = df
        self.cols = cols
        self.irradiance_threshold_noon = irradiance_threshold_noon
        self.irradiance_threshold_night = irradiance_threshold_night
        self.max_irradiance = max_irradiance
        self.base_load_correction_factor = base_load_correction_factor
        # why higher threshold_noon?
        # because the when PV gen is dominating, the error is smaller.
        self.base_load = None
        self.correction_factors = None

    def estimate_capacity(self) -> pd.Series:
        self.base_load_estimation()
        self.noon_load_estimation()
        # first use base_load - peak_load, then use the correction factor
        pv_gen_not_corrected = self.base_load - \
            self.high_irradiance_df[self.cols]
        pv_gen_corrected = pv_gen_not_corrected.multiply(
            self.correction_factors, axis=0)
        pv_gen_capacity = pv_gen_corrected.mean()
        return pv_gen_capacity

    def base_load_estimation(self):
        # use the net load at night time as the base load
        # when ghi is below threshold, the net load is the base load
        self.base_load_at_night = self.df[self.df['ghi']
                                          < self.irradiance_threshold_night][self.cols].mean()
        # calculate the correction factor based on total_con(at daylight)/total_con(at night)
        if self.base_load_correction_factor is None:
            self.base_load_correction_factor = self.df[self.df['ghi'] > self.irradiance_threshold_noon]['total_con'].mean(
            ) / self.df[self.df['ghi'] <= self.irradiance_threshold_night]['total_con'].mean()
        self.base_load = self.base_load_at_night * self.base_load_correction_factor
        return None

    def noon_load_estimation(self):
        # add a correction factor here, which is max_irradiance/ghi
        self.high_irradiance_df = self.df[self.df['ghi']
                                          > self.irradiance_threshold_noon].copy()
        self.correction_factors = self.max_irradiance / \
            self.high_irradiance_df['ghi']
        return None


def sensitivity_analysis_irradiance_threshold(df_tmp: pd.DataFrame, real_capacity_tmp: np.ndarray, fig_name: str, base_load_correction_factor: float = None, metric: str = 'PE'):
    """
    Performs sensitivity analysis on ghi thresholds and creates a heatmap visualization.

    This function analyzes how different combinations of day and night ghi thresholds
    affect the capacity estimation accuracy. The values of the heatmap are given by the specified metric.

    Parameters:
    -----------
    df_tmp : pandas.DataFrame
        DataFrame containing the power and ghi measurements.
    real_capacity_tmp : numpy.ndarray or array-like
        Array of actual PV system capacities.
    fig_name : str
        File path where the resulting heatmap will be saved.
    base_load_correction_factor : float, optional
        Correction factor for base load estimation. Default is None.
    metric : str, optional
        Performance metric to use. Options are:
        - 'PE': Percentage Error
        - 'MPE': Mean Percentage Error
        - 'R2': R-squared score
        Default is 'PE'.

    Returns:
    None
    """
    list_irradiance_thresholds_noon = [
        10, 20, 30, 40] + list(range(50, 550, 50))
    list_irradiance_thresholds_night = [0.01, 0.1, 1, 10]
    net_cols_without_total_net = [
        col for col in df_tmp.columns if col.endswith('_net') and col != 'total_net']
    # store the error rates in a matrix
    error_rates = np.zeros(
        (len(list_irradiance_thresholds_noon), len(list_irradiance_thresholds_night)))
    for i, irradiance_threshold_noon in enumerate(list_irradiance_thresholds_noon):
        for j, irradiance_threshold_night in enumerate(list_irradiance_thresholds_night):
            cap_est_based_load = capacity_estimation_base_load(
                df_tmp, net_cols_without_total_net, irradiance_threshold_noon, irradiance_threshold_night, base_load_correction_factor=base_load_correction_factor)
            capacity_based_load = cap_est_based_load.estimate_capacity()
            if metric == 'PE':
                error_rates[i, j] = calc_percentage_error(
                    real_capacity_tmp, capacity_based_load)
            elif metric == 'MPE':
                error_rates[i, j] = calc_mean_percentage_error(
                    real_capacity_tmp, capacity_based_load)
            elif metric == 'R2':
                error_rates[i, j] = calc_r2_score(
                    real_capacity_tmp, capacity_based_load)
    if metric == 'R2':
        ticks = np.linspace(-1, 1, 11)
        v_min_n_max = [-1, 1]
        cmap = 'rocket'
        label = '$\mathrm{R}^2$'
    if metric == 'PE':
        ticks = np.linspace(-100, 100, 11)
        v_min_n_max = [-100, 100]
        cmap = 'RdBu_r'
        label = 'Percentage Error ($\%$)'
    if metric == 'MPE':
        ticks = np.linspace(-100, 100, 11)
        v_min_n_max = [-100, 100]
        cmap = 'RdBu_r'
        label = 'Mean Percentage Error ($\%$)'
    # Plot heatmap
    plt.style.use(['science'])
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        error_rates,
        cmap=cmap,
        center=0,
        linewidths=0.1,
        vmin=v_min_n_max[0],
        vmax=v_min_n_max[1],
        xticklabels=list_irradiance_thresholds_night,
        yticklabels=list_irradiance_thresholds_noon,
        annot=True,
        cbar_kws={
            'label': label,
            'ticks': ticks
        }
    )
    # Configure axis labels
    plt.xlabel('$I_{night}$ (W/m$^2$)')
    plt.ylabel('$I_{day}$ (W/m$^2$)')
    # Rotate y-axis labels to horizontal
    plt.yticks(rotation=0)
    # Save and display
    plt.tight_layout()
    plt.savefig(fig_name, format='pdf')
    plt.show()
    return None
