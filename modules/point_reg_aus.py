import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import seaborn as sns
from modules.nn_regressors_aus import FCNNRegressor, ResNetRegressor, LSTMRegressor
import scienceplots
from matplotlib.dates import DateFormatter, DayLocator
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


class point_regressor():
    def __init__(self, original_data: pd.DataFrame, list_of_features: list, y_col_name: str = 'total_pv_gen_normalized', random_seed: int = 42, ghi_threshold: float = 0.01):
        self.list_of_features = list_of_features
        self.random_seed = random_seed
        self.data = original_data
        self.ghi_threshold = ghi_threshold
        self.y_col_name = y_col_name
        self.selected_features = None
        self.y_pred_mlr = None
        self.y_pred_rfr = None
        self.y_pred_lgb = None
        self.y_pred_fcnn = None
        self.y_pred_resnet = None
        self.y_pred_lstm = None
        self.res_mlr = None
        self.res_rfr = None
        self.res_lgb = None
        self.res_fcnn = None
        self.res_resnet = None
        self.res_lstm = None
        self.list_to_plot = [
            self.y_col_name,
            f'{self.y_col_name}_predicted_mlr',
            f'{self.y_col_name}_predicted_rfr',
            f'{self.y_col_name}_predicted_lgb',
            f'{self.y_col_name}_predicted_fcnn',
            f'{self.y_col_name}_predicted_resnet',
            f'{self.y_col_name}_predicted_lstm',
        ]
        self.split_data()
        plt.style.use(['science'])

    def split_data(self):
        # It is essential to have exchangeability between calibration and test set
        # when Conformal Prediction is needed, use the following split
        #####################################
        # 2010-07-01 to 2011-07-01
        self.train_data = self.data[(self.data['datetime'] >= '2012-07-01')
                                    & (self.data['datetime'] <= '2013-05-01')]
        # 2018-09-23 to 2018-12-22 as calibration
        # autumn equinox to winter solstice
        self.cal_data = self.data[(self.data['datetime'] >= '2013-05-01') &
                                  (self.data['datetime'] <= '2013-07-01')]
        # 2018-12-23 to 2019-03-10 06:00:00 as test
        # winter solstice to spring equinox
        self.test_data = self.data[(self.data['datetime'] >= '2013-05-01')
                                   & (self.data['datetime'] <= '2013-07-01')]

        # Filter data where ghi is greater than or equal to ghi_threshold
        self.train_data = self.train_data[self.train_data['ghi']
                                          >= self.ghi_threshold]
        self.cal_data = self.cal_data[self.cal_data['ghi']
                                      >= self.ghi_threshold]
        self.test_data = self.test_data[self.test_data['ghi']
                                        >= self.ghi_threshold]

        self.X_prop_train = self.train_data[self.list_of_features]
        self.X_cal = self.cal_data[self.list_of_features]
        self.X_test = self.test_data[self.list_of_features]
        self.X_train = np.concatenate([self.X_prop_train, self.X_cal], axis=0)
        self.y_prop_train = self.train_data[self.y_col_name]
        self.y_cal = self.cal_data[self.y_col_name]
        self.y_test = self.test_data[self.y_col_name]
        self.y_train = np.concatenate([self.y_prop_train, self.y_cal], axis=0)
        # convert y_test to numpy array
        self.y_test = np.array(self.y_test)
        # check shapes
        print(f'X_train.shape: {self.X_train.shape}, X_test.shape: {self.X_test.shape}, X_prop_train.shape: {self.X_prop_train.shape}, X_cal.shape: {self.X_cal.shape}')

    def use_all_features(self):
        self.use_selected_features(self.list_of_features)

    def use_selected_features(self):
        self.use_certain_features(self.selected_features)
        print(f'Using selected features: {self.selected_features}')

    def use_certain_features(self, certain_features: list):
        self.X_prop_train = self.X_prop_train[certain_features]
        self.X_cal = self.X_cal[certain_features]
        self.X_test = self.X_test[certain_features]
        self.X_train = np.concatenate([self.X_prop_train, self.X_cal], axis=0)

    def sequential_selection(self, direction: str = 'forward', n_features: int = 4):
        rfr = RandomForestRegressor(
            n_jobs=-1, n_estimators=128, random_state=self.random_seed)
        # Define the feature selector
        sfs = SequentialFeatureSelector(rfr,
                                        n_features_to_select=n_features,
                                        n_jobs=-1,
                                        direction=direction,
                                        cv=5,
                                        scoring='neg_root_mean_squared_error')
        sfs.fit(self.X_train, self.y_train)
        # print get support
        support_mask = sfs.get_support()
        print(f'{direction} selection: {support_mask}')
        # Convert list to numpy array for boolean indexing
        features_selected = np.array(self.list_of_features)[support_mask]
        print(
            f'{direction} selected features: {features_selected}')

        # if direction is forward,
        if direction == 'forward':
            self.forward_selected_features = features_selected
        # if direction is backward
        if direction == 'backward':
            self.backward_selected_features = features_selected
        # if both forward and backward selected features are the same
        if hasattr(self, 'forward_selected_features') and hasattr(self, 'backward_selected_features'):
            if np.array_equal(self.forward_selected_features, self.backward_selected_features):
                self.selected_features = self.forward_selected_features
        return None

    def sequential_selection_mlxtend(self, direction: str = 'forward', n_features: int = 4):
        rfr = RandomForestRegressor(
            n_jobs=-1, n_estimators=100, random_state=self.random_seed)

        # Define the feature selector
        sfs = SFS(rfr,
                  k_features=n_features,
                  n_jobs=-1,
                  forward=True if direction == 'forward' else False,
                  scoring='r2',
                  cv=5,
                  verbose=2)
        sfs.fit(self.X_train, self.y_train)
        # print get support
        support_mask = sfs.k_feature_idx_
        print(f'{direction} selection: {support_mask}')
        # use the support mask to get the features
        features_selected = [self.list_of_features[i] for i in support_mask]
        print(
            f'{direction} selected features: {features_selected}')

        # if direction is forward,
        if direction == 'forward':
            self.forward_selected_features = features_selected
        # if direction is backward
        if direction == 'backward':
            self.backward_selected_features = features_selected
        # if both forward and backward selected features are the same
        if hasattr(self, 'forward_selected_features') and hasattr(self, 'backward_selected_features'):
            if np.array_equal(self.forward_selected_features, self.backward_selected_features):
                self.selected_features = self.forward_selected_features

        print("\nAll Feature Subsets Evaluated:")
        for i, feature_subset in enumerate(sfs.subsets_.values(), start=1):
            print(f"Iteration {i}:")
            print(f"Selected Features: {feature_subset['feature_idx']}")
            print(f"Performance: {feature_subset['avg_score']}")
        # plot results of sfs
        fig = plot_sfs(sfs.get_metric_dict(),
                       kind='std_dev',
                       figsize=(6, 4),
                       ylabel=r'$\mathrm{R}^2$')
        plt.title(f'Sequential {direction.capitalize()} Selection (w. StdDev)')
        plt.grid()
        plt.savefig(
            f'../figs/sequential_{direction}_selection_w_stddev.pdf', format='pdf')
        plt.show()
        return None

    def remove_suffix_norm(self, labels: list):
        # remove suffix "_norm" from the label
        return [label.replace('_norm', '') for label in labels]

    def tree_based_selection_MDI(self):
        # mean decrease in impurity
        rf = self.rf_fitted_on_prop_train()
        importances = rf.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in rf.estimators_], axis=0)
        forest_importances = pd.Series(
            importances, index=self.list_of_features)
        # sort the forest_importances by the importances
        forest_importances = forest_importances.sort_values(ascending=True)

        # Set figure size
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot the horizontal bar chart
        # Use barh for horizontal bars
        forest_importances.plot.barh(xerr=std, ax=ax,  alpha=0.8, error_kw={
                                     'ecolor': 'black', 'capsize': 5, 'capthick': 1.5})
        # for the y-axis ticks, remove suffix of "_norm"
        # Get current tick labels
        labels = [label.get_text() for label in ax.get_yticklabels()]
        # Remove '_norm' from each label
        new_labels = self.remove_suffix_norm(labels)
        # Set the new labels
        ax.set_yticklabels(new_labels)

        # Set title and labels with increased font sizes
        ax.set_title("Feature importances using MDI",
                     fontsize=16)  # Title font size
        ax.set_xlabel("Mean decrease in impurity",
                      fontsize=14)  # X-axis label font size
        ax.set_ylabel("Features", fontsize=14)  # Y-axis label font size
        ax.tick_params(axis='x', labelsize=12)  # X-axis tick labels font size
        ax.tick_params(axis='y', labelsize=12)  # Y-axis tick labels font size

        fig.tight_layout()  # Adjust layout to avoid overlap
        fig.savefig(
            '../figs/feature_importance_MDI.pdf', format='pdf')

    def tree_based_selection_permutation(self):
        # Fit the random forest model on training data
        rf = self.rf_fitted_on_prop_train()

        # Perform permutation importance
        result = permutation_importance(
            estimator=rf,
            X=self.X_train,
            y=self.y_train,
            n_repeats=10,
            random_state=self.random_seed,
            n_jobs=-1
        )

        # Create a pandas Series for the importances
        forest_importances = pd.Series(
            result.importances_mean, index=self.list_of_features)
        # sort the forest_importances by the importances
        forest_importances = forest_importances.sort_values(ascending=True)

        # Set figure size
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot the horizontal bar chart
        forest_importances.plot.barh(
            xerr=result.importances_std, ax=ax,  alpha=0.8, error_kw={
                'ecolor': 'black', 'capsize': 5, 'capthick': 1.5}
        )
        # for the y-axis ticks, remove suffix of "_norm"
        # Get current tick labels
        labels = [label.get_text() for label in ax.get_yticklabels()]
        # Remove '_norm' from each label
        new_labels = self.remove_suffix_norm(labels)
        # Set the new labels
        ax.set_yticklabels(new_labels)
        # Set the title and labels with adjusted font sizes
        ax.set_title(
            "Feature importances using feature permutation", fontsize=16)
        # X-axis now represents the importance
        ax.set_xlabel("Mean accuracy decrease", fontsize=14)
        # Y-axis represents the feature names
        ax.set_ylabel("Features", fontsize=14)
        ax.tick_params(axis='x', labelsize=12)  # Adjust X-axis tick label size
        ax.tick_params(axis='y', labelsize=12)  # Adjust Y-axis tick label size

        # Adjust layout and show the plot
        fig.tight_layout()
        fig.savefig(
            '../figs/feature_importance_permutation.pdf', format='pdf')

    def rf_fitted_on_prop_train(self):
        rf = RandomForestRegressor(
            n_jobs=-1, n_estimators=100, random_state=self.random_seed)
        rf.fit(self.X_prop_train, self.y_prop_train)
        return rf

    def mlr_fitted_on_prop_train(self):
        mlr = LinearRegression()
        mlr.fit(self.X_prop_train, self.y_prop_train)
        return mlr

    def xgb_fitted_on_prop_train(self):
        xgb = XGBRegressor(random_state=self.random_seed)
        xgb.fit(self.X_prop_train, self.y_prop_train)
        return xgb

    def lightgbm_fitted_on_prop_train(self):
        lgb = LGBMRegressor(random_state=self.random_seed)
        lgb.fit(self.X_prop_train, self.y_prop_train)
        return lgb

    def fcnn_fitted_on_prop_train(self):
        reg = FCNNRegressor(random_state=self.random_seed)
        reg.fit(self.X_prop_train, self.y_prop_train)
        return reg

    def resnet_fitted_on_prop_train(self):
        reg = ResNetRegressor(random_state=self.random_seed)
        reg.fit(self.X_prop_train, self.y_prop_train)
        return reg

    def lstm_fitted_on_prop_train(self):
        reg = LSTMRegressor(random_state=self.random_seed)
        reg.fit(self.X_prop_train, self.y_prop_train)
        return reg

    def add_trendline(self, x, y, ax, color='red'):
        # Sort x and y values to ensure a continuous line
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Fit polynomial and create trendline
        z = np.polyfit(x_sorted, y_sorted, 2)
        p = np.poly1d(z)
        ax.plot(x_sorted, p(x_sorted), "--", color=color, linewidth=2)

    def _get_regressor_list(self):
        """List of (name, display_name, fit_method) for compare_regressors."""
        return [
            ('mlr', 'MLR', self.mlr_fitted_on_prop_train),
            ('rfr', 'RFR', self.rf_fitted_on_prop_train),
            ('lgb', 'LGB', self.lightgbm_fitted_on_prop_train),
            ('fcnn', 'FCNN', self.fcnn_fitted_on_prop_train),
            ('resnet', 'ResNet', self.resnet_fitted_on_prop_train),
            ('lstm', 'LSTM', self.lstm_fitted_on_prop_train),
        ]

    def compare_regressors(
        self,
        fname1: str = "regressor_comparison1.pdf",
        fname2: str = "regressor_comparison2.pdf",
        evaluate_on: str = "test",
    ):
        """Compare regressors. evaluate_on: 'test' (default) or 'train'."""
        self.use_selected_features()
        if evaluate_on == "test":
            X_eval = self.X_test
            y_eval = self.y_test
            eval_label = "test"
        elif evaluate_on == "train":
            X_eval = self.X_prop_train
            y_eval = np.array(self.y_prop_train)
            eval_label = "train"
        else:
            raise ValueError("evaluate_on must be 'test' or 'train'")
        print(f"Evaluating on {eval_label} set (n={len(y_eval)} samples).")
        reg_list = self._get_regressor_list()
        n_reg = len(reg_list)

        # Fit, predict, and compute metrics for each regressor
        models = []
        r2_list, adj_r2_list, rmse_list = [], [], []
        train_time_list = []
        y_pred_list, res_list = [], []

        for name, display_name, fit_method in reg_list:
            t0 = time.perf_counter()
            model = fit_method()
            train_time_list.append(time.perf_counter() - t0)
            models.append((name, model))
            y_pred = model.predict(X_eval)
            y_pred_list.append(y_pred)
            res_list.append(y_eval - y_pred)
            r2, adj_r2 = self.r2_score_actual_vs_predicted(y_eval, y_pred)
            rmse = self.root_mean_squared_error(y_eval, y_pred)
            r2_list.append(r2)
            adj_r2_list.append(adj_r2)
            rmse_list.append(rmse)

        # Performance table
        performance_table = pd.DataFrame({
            'R2': r2_list,
            'Adjusted R2': adj_r2_list,
            'RMSE': rmse_list,
            'Training time (s)': train_time_list,
        }, index=[r[0] for r in reg_list])
        print(performance_table)
        performance_table.to_csv(
            '../figs/point_reg_performance_table.csv', index=True)

        # Figure 1: actual vs predicted (1 row x n_reg)
        fig1, axes1 = plt.subplots(
            1, n_reg, figsize=(4 * n_reg, 4), sharey=True)
        if n_reg == 1:
            axes1 = [axes1]
        for ax, y_pred, (_, display_name, _) in zip(axes1, y_pred_list, reg_list):
            ax.scatter(x=y_eval, y=y_pred, alpha=0.5)
            ax.plot(y_eval, y_eval, 'k-', linewidth=2)
            self.add_trendline(y_eval, y_pred, ax)
            ax.set_title(display_name)
            ax.set_aspect('equal')
            ax.set_xlim(y_eval.min(), y_eval.max())
            ax.set_ylim(y_pred.min(), y_pred.max())
        axes1[0].set_ylabel('Predicted values', fontsize=16)
        axes1[n_reg // 2].set_xlabel('Actual values', fontsize=16)
        for ax in axes1:
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
        fig1.savefig(f'../figs/{fname1}', format='pdf')

        # Figure 2: residuals (1 row x n_reg)
        fig2, axes2 = plt.subplots(
            1, n_reg, figsize=(4 * n_reg, 4), sharey=True)
        if n_reg == 1:
            axes2 = [axes2]
        x_max = y_eval.max()
        x_min = min(y_eval.min(), -0.2)
        x_range = x_max - x_min
        for ax, res, (_, display_name, _) in zip(axes2, res_list, reg_list):
            res_arr = np.asarray(res)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(res_arr.min(), res_arr.max())
            # Distribution of residuals inside plot (left side), numpy histogram only
            density, bin_edges = np.histogram(res_arr, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = 0.12 * x_range * (density / (density.max() + 1e-12))
            ax.fill_betweenx(bin_centers, x_min, x_min + width, color='C0', alpha=0.5, zorder=0)
            ax.plot(x_min + width, bin_centers, color='C0', lw=1, zorder=0)
            ax.scatter(x=y_eval, y=res, alpha=0.5, zorder=1)
            self.add_trendline(y_eval, res, ax)
            ax.plot(y_eval, np.zeros(len(y_eval)), 'k-', linewidth=2)
            ax.set_title(display_name)
        axes2[0].set_ylabel('Residuals', fontsize=16)
        axes2[n_reg // 2].set_xlabel('Actual values', fontsize=16)
        for ax in axes2:
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14, labelleft=True)
        fig2.tight_layout()
        fig2.savefig(f'../figs/{fname2}', format='pdf')
        plt.show()

        # Store predictions and residuals on self
        for (name, _, _), y_pred, res in zip(reg_list, y_pred_list, res_list):
            setattr(self, f'y_pred_{name}', y_pred)
            setattr(self, f'res_{name}', res)

        # Full-data predictions and zero below ghi_threshold
        for name, model in models:
            col = f'{self.y_col_name}_predicted_{name}'
            self.data[col] = model.predict(self.data[self.selected_features])
            self.data.loc[self.data['ghi'] < self.ghi_threshold, col] = 0

    def r2_score_actual_vs_predicted(self, y_test, y_pred) -> None:
        r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = self.X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return r2, adj_r2

    def root_mean_squared_error(self, y_test, y_pred) -> None:
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        return rmse

    def df_common_xylabel_plot(self, df, y, doy_start=0, doy_end=366, year=[2017, 2018, 2019], xlabel='Datetime', ylabel="example ylabel", layout=(7, 4), subplots=True, figsize=(8, 4), legends=[], fname=None):
        # Filter data
        filtered_df = df[(df['doy'] >= doy_start) & (
            df['doy'] <= doy_end) & (df['year'].isin(year))]
        # convert datetime to datetime object
        filtered_df.loc[:, 'datetime'] = pd.to_datetime(
            filtered_df['datetime'])

        if subplots:
            fig, axes = plt.subplots(
                layout[0], layout[1], figsize=figsize, sharex=True)
            axes = axes.flatten()

            for idx, col in enumerate(y):
                ax = axes[idx]
                ax.xaxis.axis_date('UTC')
                ax.plot(filtered_df['datetime'], filtered_df[col])
                ax.set_title(col)
                ax.xaxis.set_major_formatter(DateFormatter('%d-%H'))
                # Add major ticks at midnight (00:00) for each day
                ax.xaxis.set_major_locator(DayLocator())
                ax.tick_params(axis='both', which='major')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            for col in y:
                ax.xaxis.axis_date('UTC')
                ax.plot(filtered_df['datetime'], filtered_df[col], label=col)
            if legends:
                ax.legend(legends)
            else:
                ax.legend()
            # format yyyy-mm-dd
            ax.xaxis.set_major_formatter(DateFormatter('%d-%H'))
            # Add major ticks at midnight (00:00) for each day
            ax.xaxis.set_major_locator(DayLocator())
            ax.tick_params(axis='both', which='major')
        # Add common labels
        fig.text(0.05, 0.5, ylabel, va='center', rotation='vertical')
        fig.text(0.5, 0.0, xlabel, ha='center')

        # save the plot
        if fname:
            fig.savefig(
                f'../figs/{fname}', format='pdf')

    def visualize_results_by_dates_2011_Jan_7days(self, legends=[], fname="tmp"):
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2011], doy_start=1, doy_end=7, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)

    def visualize_results_by_dates_2012_Jan_7days(self, legends=[], fname="tmp"):
        self.df_common_xylabel_plot(self.data, y=self.list_to_plot, year=[
            2012], doy_start=1, doy_end=7, ylabel='Normalized PV Generation', layout=(1, 1), figsize=(8, 4), subplots=False, legends=legends, fname=fname)
