import pandas as pd
import numpy as np
import pvlib
from sklearn.preprocessing import MinMaxScaler
import os


class data_preprocessor():
    def __init__(self, data):
        self.data = data
        self.pv_data = None
        self.con_data = None
        self.net_data = None

        # Read column lists from files
        self.time_columns = self._read_column_list('time_columns.txt')
        self.weather_columns = self._read_column_list(
            'weather_columns_preprocess.txt')
        self.generate_pv_data()
        self.generate_con_data()
        self.generate_net_data()

    def _read_column_list(self, filename):
        """Read column list from config file"""
        config_dir = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'config')
        filepath = os.path.join(config_dir, filename)

        with open(filepath, 'r') as f:
            columns = [line.strip() for line in f.readlines() if line.strip()]
        return columns

    def generate_pv_data(self):
        # keep the columns that contain _pv and the columns of time, weather
        pv_columns = self.data.columns[self.data.columns.str.contains(
            '_pv')].tolist()
        all_columns = pv_columns + self.time_columns + self.weather_columns
        self.pv_data = self.data[all_columns]
        # remove the _pv suffix from the columns
        self.pv_data.columns = self.pv_data.columns.str.replace('_pv', '')

    def generate_con_data(self):
        con_columns = self.data.columns[self.data.columns.str.contains(
            '_con')].tolist()
        all_columns = con_columns + self.time_columns + self.weather_columns
        self.con_data = self.data[all_columns]
        # remove the _con suffix from the columns
        self.con_data.columns = self.con_data.columns.str.replace('_con', '')

    def generate_net_data(self):
        net_columns = self.data.columns[self.data.columns.str.contains(
            '_net')].tolist()
        all_columns = net_columns + self.time_columns + self.weather_columns
        self.net_data = self.data[all_columns]
        # remove the _net suffix from the columns
        self.net_data.columns = self.net_data.columns.str.replace('_net', '')

    def estimate_pv_capacity_from_pv_profiles_avg_high_values(self):
        # drop the columns of time, weather, and weather normalized
        time_columns = self.time_columns
        weather_columns = self.weather_columns

        # Only drop columns that exist in the DataFrame
        columns_to_drop = [col for col in time_columns +
                           weather_columns if col not in 'ghi' if col in self.pv_data.columns]
        df = self.pv_data.drop(columns=columns_to_drop)
        # can not be too high, otherwise no data sample for some households.
        irradiance_th = 500
        irradiance_max = 1000
        # get the rows where the irradiance is greater than the threshold
        pv_gen_high_irradiance = df[df['ghi'] > irradiance_th]
        # rescale the pv_gen_high_irradiance by 1000/irradiance
        pv_gen_high_irradiance = pv_gen_high_irradiance.multiply(
            irradiance_max/pv_gen_high_irradiance['ghi'], axis=0)
        # remove the irradiance column
        pv_gen_high_irradiance = pv_gen_high_irradiance.drop(
            columns=['ghi'])
        # take the mean of the rescaled pv_gen_high_irradiance
        self.pv_capacity_from_pv_profiles = pv_gen_high_irradiance.mean()

    def normalize_pv(self) -> None:
        # normalize the PV generation of each household by the its installed capacity
        # by dividing all the xxxxx_pv columns with its maximum value
        pv_columns = [
            col for col in self.data.columns if col.endswith('_pv')]
        df_normalized = self.data.copy()
        df_normalized[pv_columns] = df_normalized[pv_columns].div(
            df_normalized[pv_columns].max())
        self.data_pv_normalized = df_normalized
        self.pv_normalized = True
        return df_normalized

    def add_total_net(self):
        net_columns = [
            col for col in self.data.columns if col.endswith('_net')]
        self.data['total_net'] = self.data[net_columns].sum(axis=1)
        self.data_pv_normalized['total_net'] = self.data['total_net']
        self.total_net_added = True
        return None

    def add_total_con(self):
        con_columns = [
            col for col in self.data.columns if col.endswith('_con')]
        self.data['total_con'] = self.data[con_columns].sum(axis=1)
        self.data_pv_normalized['total_con'] = self.data['total_con']
        self.total_con_added = True
        return None

    def add_total_pv_gen(self):
        # add a column of "total_pv_gen"
        # do not use the normalized pv data!
        pv_columns = [
            col for col in self.data.columns if col.endswith('_pv')]
        self.data['total_pv_gen'] = self.data[pv_columns].sum(
            axis=1)
        self.data_pv_normalized['total_pv_gen'] = self.data['total_pv_gen']
        self.total_pv_gen_added = True
        return None

    def normalize_feature(self, feature_list: list):
        scaler = MinMaxScaler()
        feature_list_norm = [col + '_norm' for col in feature_list]
        self.data_pv_normalized[feature_list_norm] = scaler.fit_transform(
            self.data_pv_normalized[feature_list])
        return None

    def add_total_pv_gen_normalized(self, capacity: pd.Series):
        # default capacity is self.pv_capacity_from_pv_profiles
        # only when the household is generating pv power, its capacity is added to the total capacity.
        # For each row, the "normalized_total_pv_gen_correct" would be
        # (sum of pv generation)/(installed capacity of the households that has pv gen not NaN)
        df_tmp = self.data.copy()
        # active_installed_capacity is the sum of installed capacity of the households that has pv gen not NaN
        # initialize active_installed_capacity, with the shape (len(df_tmp),)
        active_installed_capacity = np.zeros(len(df_tmp))
        # loop over each row to get the active_installed_capacity
        for index, row in df_tmp.iterrows():
            # check the pv_columns if the pv generation is not NaN, get the name of the columns
            pv_columns = [col for col in df_tmp.columns if col.endswith('_pv')]
            row_pv_columns = row[pv_columns]
            # for the row variable, get the index of those not NaN
            houses_not_nan = row_pv_columns[row_pv_columns.notna()].index
            # remove the "_pv" from the names in houses_not_nan
            houses_not_nan = [col.replace('_pv', '') for col in houses_not_nan]
            # Use houses_not_nan to get the installed capacity from df_installed_capacity
            active_installed_capacity[index] = capacity.loc[houses_not_nan].sum(
            )
        # add a column of "total_pv_gen_normalized"
        df_tmp['total_pv_gen_normalized'] = df_tmp['total_pv_gen'] / \
            active_installed_capacity
        self.data_pv_normalized['total_pv_gen_normalized'] = df_tmp['total_pv_gen_normalized']
        self.total_pv_gen_normalized_added = True
        return None

    def add_cos_sin_HoD(self):
        self.data_pv_normalized['cos_HoD'] = np.cos(
            2*np.pi*self.data_pv_normalized['HoD']/24)
        self.data_pv_normalized['sin_HoD'] = np.sin(
            2*np.pi*self.data_pv_normalized['HoD']/24)
        return None
