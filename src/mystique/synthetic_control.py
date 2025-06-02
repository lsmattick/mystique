"""Class for Synthetic Control."""
from datetime import datetime
import logging
from typing import Callable, Dict, List, Tuple, Union

import colorful as colors
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from .utils import calculate_mspe
from .base import MystiqueBase
from mystique import rust


class SyntheticControl(MystiqueBase):
    """Construct counterfactual data for experiments to estimate treatment effect.

    The terms "counterfactual" and "synthetic control" are used similarly in this class.
    "Donor pool" and "control units" are also used similarly.

    This algo generates counterfactual data for an experiment given the time series data for the
    test units and the control units. This is done by finding a set of weights W for the control
    units that minimizes the following, for J control units, K test units and m features.

    ||X_i - X_0 * W_i||_V + lambda_i * SUM_j w_j||X_i - X_j||

    Refer to the Mystique readme for more information on this problem.

    Instance attributes:

    data: pd.DataFrame
        Pandas data frame containing the time series data for the outcome variable and any
        covariates to be used.

    test_units: List[str],
        List of units that received treatment.
        e.g. ["campaign1", "campaign2"]

    units_col: str,
        Column name from data frame provided in "data" argument above to look up test units and
        control units.

    outcome_metric: str,
        Column name from data frame provided in "data" argument above that contains the outcome
        metric of interest.

    date_col: str,
        Column name from data frame provided in "data" argument above that contains the date.

    event_date: Union[str, None] = None,
        Date "YYYY-MM-DD" that marks when the treatment started. This date will be included in the
        pre intervention data. If event date is left as None, it will default to the max date found
        in the data.

    covariates: Union[List[str], None] = None,
        Default to []. List of columns that should be included as features in the synthetic control
        algorithm.

    model_type: Union[str, None] = None,
        Default to prospective. There are two model types, prospective and retrospective.

        Prospective: The penalty term lambda is trained by splitting the pre intervention data into
        a test set and a train set.

        Retrospective: The penalty term lambda is trained by cross validation on the control units
        using both pre and post intervention data.

    norm_type: Union[str, None] = None,
        Default to induced. There are two norm types, euclidean and induced.

        Euclidean: Standard euclidean norm from np.linalg.norm

        Induced: The norm induced by the diagonal matrix V whose diagonal consists of weights which
        estimate the predictive power of the features used in the minimization problem.

    lam: Union[float, None] = None,
        Short for lambda. Providing a value for lam will skip the training process for lambda and
        use whatever is provided here.

    training_periods: Union[int, None] = None,
        For the prospective model type, the number of training periods to use. If nothing is
        provided, defaults to a default split of 80 20.

    aggregate_function_map: Union[dict, None] = None,
        Dict where the keys are strings, column names, that need to be calculated and the values are
        the functions that should be used to calculate them. This is necessary if the outcome metric
        provided cannot be aggregated by summing. For example, revenue, spend, profit, sessions can
        all be summed
            aggregate_function_map = {
                "ctr": lambda df: df["clicks"] / df["sessions"],
                "rpc": lambda df: df["revenue"] / df["clicks"]
            }

    max_attempts: Union[int, None] = None,
        Default to 200. Maximum number of attempts to minimize the main cost functions using scipy.
        Scipy does not always complete successfully. It may either return the starting value, or
        fail to minimize completely, which warrants multiple attempts. After 100 attempts, switch
        from using slsqp to trust-constr.
    """

    valid_model_types = ["prospective", "retrospective"]

    valid_norm_types = ["euclidean", "induced"]

    train_pct = 0.8
    max_lambda = 100
    default_model_type = "prospective"
    default_norm_type = "induced"
    default_max_attempts = 200

    def __init__(
        self,
        data: pd.DataFrame,
        test_units: List[str],
        units_col: str,
        outcome_metric: str,
        date_col: str,
        event_date: Union[str, None] = None,
        covariates: Union[List[str], None] = None,
        model_type: Union[str, None] = None,
        norm_type: Union[str, None] = None,
        lam: Union[float, None] = None,
        training_periods: Union[int, None] = None,
        aggregate_function_map: Union[dict, None] = None,
        max_attempts: Union[int, None] = None,
    ):
        self.data = data
        self.test_units = test_units
        self.units_col = units_col
        self.outcome_metric = outcome_metric
        self.event_date = event_date
        self.date_col = date_col
        self.covariates = covariates or []
        self.model_type = model_type or self.default_model_type
        self.norm_type = norm_type or self.default_norm_type
        self.lam = lam
        self.training_periods = training_periods
        self.aggregate_function_map = aggregate_function_map
        self.max_attempts = max_attempts or self.default_max_attempts

        self._validate_event_date_format()
        self._validate_columns()

        self._validate_model_type()
        self._validate_norm_type()
        self._validate_max_attempts()

        self.data = self._preprocess_data(self.data)

        self._validate_test_units()
        self._validate_event_date_in_range()
        self._validate_data_pre_event_date()
        self._validate_model_type_against_event_date()

        self.pre_int_periods = self._calculate_pre_int_periods()

        self._validate_training_periods()

        self.train_test_split_date = self._get_train_test_split_date()
        self.event_date = self._coalesce_event_date()
        self.control_units = self._get_control_units()

        self.out_cols = [self.units_col, self.outcome_metric]
        self.out_cols += [var for var in self.covariates if var not in self.out_cols]

        self.post_int_exists = self._check_post_int_exists()

    def log_info(self, placebo: bool, msg: str):
        """Wrapper for logging, only log if placebo is False."""
        if not placebo:
            logging.info(msg)

    def getargs(self, local_vars: dict, args: List[str]) -> tuple:
        """Parse key, vals in local_vars.items() and update None's with getattr(self, key)."""
        return_vars = dict()
        for key, val in local_vars.items():
            if key != "self":
                return_vars[key] = val if val is not None else getattr(self, key)
        if len(args) == 1:
            return return_vars[args[0]]
        else:
            return tuple([return_vars[arg] for arg in args])

    def _validate_model_type(self):
        msg = (
            "Model type provided is not supported.\n"
            f"Please use one of the supported model types: {self.valid_model_types}"
        )
        assert self.model_type in self.valid_model_types, msg

    def _validate_norm_type(self):
        msg = (
            "Norm type provided is not supported.\n"
            f"Please use one of the supported norm types: {self.valid_norm_types}"
        )
        assert self.norm_type in self.valid_norm_types, msg

    def _validate_columns(self):
        cols = [self.date_col, self.units_col, self.outcome_metric] + self.covariates
        cols_not_found = list(set(cols) - set(self.data.columns))
        msg = f"Some of the column names provided were not found data: {cols_not_found}"
        assert len(cols_not_found) == 0, msg

    def _validate_max_attempts(self):
        assert self.max_attempts > 0

    def _validate_model_type_against_event_date(self):
        if self.model_type == "retrospective":
            msg = (
                "event_date is None, event_date is required for retrospective modeling."
            )
            assert self.event_date is not None, msg

            if self.event_date is not None:
                max_date = self.data[self.date_col].max()
                event_datetime = datetime.strptime(self.event_date, "%Y-%m-%d")
                msg = (
                    "event_date cannot be the max date. Retrospective modeling requires a post "
                    f"intervention period. event_date: {self.event_date}"
                )
                assert event_datetime < max_date, msg

    def _calculate_pre_int_periods(self) -> int:
        """Calculate the number of days in the pre intervention period."""
        if self.event_date is not None:
            return self.data.query(f"{self.date_col} <= '{self.event_date}'")[
                self.date_col
            ].nunique()
        else:
            return self.data[self.date_col].nunique()

    def _validate_training_periods(self):
        if self.training_periods is not None:
            msg = (
                "The number of training periods must be less than the number of pre intervention "
                "periods.\n"
                f"training periods: {self.training_periods}.\n"
                f"pre intervention periods: {self.pre_int_periods}"
            )
            assert self.training_periods < self.pre_int_periods, msg

    def _get_train_test_split_date(self):
        """Return date to split data for train test split.

        If the number of training periods is not provided, use a default of 80/20 split.
        """
        if self.model_type == "prospective":
            dates = self.data[self.date_col].unique().tolist()
            dates.sort()
            if self.training_periods is not None:
                return dates[self.training_periods - 1]
            else:
                training_periods = int(np.round(self.train_pct * self.pre_int_periods))
                return dates[training_periods - 1]
        else:
            return None

    def _get_control_units(self) -> list:
        """Return list of control units."""
        return list(set(self.data[self.units_col].unique()) - set(self.test_units))

    def _check_post_int_exists(self):
        """Return a boolean for the existence of a post intervention period."""
        return (
            datetime.strptime(self.event_date, "%Y-%m-%d")
            < self.data[self.date_col].max()
        )

    @staticmethod
    def random_guess(n: int) -> np.ndarray:
        """Return a random vector of size n of positive numbers that sum to 1.

        Arguments:
            n (int): Positive integer

        Return:
            w0 (np.ndarray): Vector of length n of positive real numbers that sum to 1.
        """
        w0 = np.random.rand(n)
        w0 = w0 / w0.sum()
        return w0

    @staticmethod
    def norm_x_over_V(x: np.ndarray, v: np.ndarray) -> float:
        """Vector norm induced by the diagonal matrix V.

        Arguments:
            x (np.ndarray): Vector to take the norm of.

            v (np.ndarray): Vector to calculate the induced norm using the diagonal matrix
            v * I, where I is the identity matrix.

        Return:
            (float) The norm of x.
        """
        V = v * np.identity(v.size)
        return np.sqrt(np.dot(np.dot(x.T, V), x)).flat[0]

    def norm_factory(self) -> Callable[[np.ndarray], float]:
        """Return either the euclidean norm, or an induced norm depending on norm type.

        Return:
            Callable[[np.ndarray], float]
            factor_weights: vector of factor weights or None
        """
        if self.norm_type == "euclidean":
            return rust.calc_norm_l2

        elif self.norm_type == "induced":

            def norm(x: np.ndarray) -> float:
                """Wrapper for norm_x_over_V."""
                return rust.norm_x_over_V(x, self.factor_weights)

            return norm

    def exclude_test_units(self, data: pd.DataFrame, test_units: list) -> pd.DataFrame:
        """Return a data frame that does not contain the provided test units.

        Arguments:
            data (pd.DataFrame): Data frame that should have test_units removed.

            test_units (list): List of test units to be removed.

        Return:
            (pd.DataFrame) The input data frame without data from the test units.
        """
        return data[~data[self.units_col].isin(test_units)]

    def join_weights_to_map(
        self, weights: np.ndarray, df_map: pd.DataFrame
    ) -> pd.DataFrame:
        """Return a data frame of weights and the corresponding control units.

        Arguments:
            weights (np.ndarray): Vector of weights.

            df_map (pd.DataFrame): Data frame with a single column of control units to which the
            weights will be joined.

        Return:
            df_weights (pd.DataFrame): Data frame of weights and corresponding control units.
        """
        df_weights = df_map.merge(
            pd.DataFrame(weights).rename(columns={0: "weight"}),
            left_index=True,
            right_index=True,
        )
        return df_weights

    def apply_aggregate_fn_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics found in self.aggregate_function_map.

        Arguments:
            df (pd.DataFrame): Data frame for which to calculate metrics.

        Return:
            df (pd.DataFrame)
        """
        logging.debug("START apply_aggregate_fn_map")

        if self.aggregate_function_map is not None:
            for metric, fn in self.aggregate_function_map.items():
                df[metric] = df.apply(fn, axis=1)

        logging.debug("END apply_aggregate_fn_map")
        return df

    def calculate_rolling_means(
        self, data: pd.DataFrame, rolling_means: list = [1, 3, 5, 7]
    ) -> pd.DataFrame:
        """Return a data frame of rolling means.

        For each rolling mean n, calculate the n day rolling mean for self.outcome_metric for the
        last n days in the input data.

        Arguments:
            data (pd.DataFrame): Data frame of time series data from which to calculate
            rolling means.

        Return:
            rolling_means (pd.DataFrame): A data frame where each of the requested rolling means are
            returned as a column f"{self.outcome_metric}_rm_{n}"} indexed by self.units_col.
        """
        logging.debug("START calculate_rolling_means")

        # Trim rolling means if necessary
        if self.training_periods is not None:
            rolling_means = [rm for rm in rolling_means if rm < self.training_periods]
        else:
            rolling_means = [rm for rm in rolling_means if rm < self.pre_int_periods]

        rm_df_list = []
        for n in rolling_means:
            # For a rolling mean of 1, return the last day of data
            if n == 1:
                last_day = (
                    data.sort_values([self.units_col, self.date_col])
                    .groupby(self.units_col)
                    .tail(n)[[self.units_col, self.outcome_metric]]
                    .rename(
                        columns={self.outcome_metric: f"{self.outcome_metric}_rm_{n}"}
                    )
                )

                rm_df_list.append(last_day)
                continue

            # Subset the last n days for each group
            last_n_days = (
                data.sort_values([self.units_col, self.date_col])
                .groupby(self.units_col)
                .tail(n)
            )

            # Sum up metrics and calculate the mean for the outcome metric. If the outcome metric
            # requires a formula, e.g. rpc or ctr, it should be included in the aggregate function
            # map which is applied in the following step.
            g = last_n_days.groupby(self.units_col, as_index=False)
            n_day_rolling_mean = g.sum(numeric_only=True).drop(
                self.outcome_metric, axis=1
            )

            mean_outcome = g.agg({self.outcome_metric: "mean"})
            n_day_rolling_mean = n_day_rolling_mean.merge(
                mean_outcome, on=self.units_col
            )

            # Calculate metrics in the aggregate function map
            n_day_rolling_mean = self.apply_aggregate_fn_map(n_day_rolling_mean)

            # Subset and rename columns
            n_day_rolling_mean = n_day_rolling_mean[
                [self.units_col, self.outcome_metric]
            ].rename(columns={self.outcome_metric: f"{self.outcome_metric}_rm_{n}"})

            rm_df_list.append(n_day_rolling_mean)

        rolling_means = rm_df_list.pop(0)

        for df in rm_df_list:
            rolling_means = rolling_means.merge(df, on=self.units_col)
        logging.debug("END calculate_rolling_means")
        return rolling_means

    def add_outcomes(
        self, pre_int_data: pd.DataFrame, time_series_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Return a data frame with additional features added as columns.

        Add the outcome for each available time period in the provided time series data frame. e.g.
        If the time series data has 10 time periods, then the 10 respective outcomes are added for
        each unit, control and test, as features.

        Arguments:
            pre_int_data (pd.DataFrame): Pre intervention data to add features to.

            time_series_data (pd.DataFrame): The time series data to get pre intervention outcomes.

        Return:
            (pd.DataFrame): pre_int_data data frame with the added outcomes.
        """
        logging.debug("START add_outcomes")

        # The input time series data is long, prepare this to pivot to wide data
        df_long = time_series_data[
            [self.date_col, self.units_col, self.outcome_metric]
        ].sort_values([self.units_col, self.date_col])
        df_long = df_long.reset_index(drop=True)
        df_long[self.date_col] = df_long[self.date_col].dt.strftime("%Y-%m-%d")

        # Pivot long data to wide. Index is now self.units_col and we get a new column for each
        # unique date with self.outcome_metric as the values. These are then merged onto the
        # pre intervention aggregate data to be treated as features.
        df_wide = df_long.pivot_table(
            index=self.units_col, columns=self.date_col, values=self.outcome_metric
        )

        df_wide = df_wide.reset_index()

        # Merge these additional features onto the pre intervention aggregate data
        pre_int_data = pre_int_data.merge(df_wide, on=self.units_col)

        logging.debug("END add_outcomes")
        return pre_int_data

    def normalize_col(
        self, data: Union[pd.Series, pd.DataFrame], method="standardization"
    ) -> pd.Series:
        """Normalize values in the column provided by dividing by the sum.

        Arguments:
            col (pd.Series): The column to normalize.

        Return:
            (pd.Series): Normalized column.
        """
        if method == "standardization":
            return self.standardize_cols(data)
        else:
            return data / data.sum()

    def standardize_cols(
        self, data: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Standardize values in the 2d array by mean and std.

        Arguments:
            data (pd.Series, pd.DataFrame): values must be in an 2d array

        Return:
            (pd.Series): Normalized column.
        """
        scaler = StandardScaler(with_mean=True, with_std=True)
        res = scaler.fit_transform(data)
        return res

    def standardize_col(self, data: Union[pd.Series, list]) -> Union[pd.Series, list]:
        """Standardize values in the column by mean and std.

        Arguments:
            data (pd.Series): value to be normalized. Data is in 1d array format

        Return:
            (pd.Series, list): Normalized value.
        """
        tmp = data.copy()
        if isinstance(tmp, pd.Series):
            tmp = tmp.values
        elif isinstance(tmp, list):
            tmp = np.array(tmp)
        if len(tmp.shape) == 1:
            tmp = tmp.reshape(-1, 1)

        scaler = StandardScaler(with_mean=True, with_std=True)
        res = scaler.fit_transform(tmp)
        return res.T[0]

    def normalize_all(
        self, data: pd.DataFrame, method: str = "standardization"
    ) -> pd.DataFrame:
        """Normalize all columns in data.

        Arguments:
            data (pd.DataFrame)

        Return:
            data (pd.DataFrame): The input data frame with all its columns normalized.
        """
        cols = data.drop(self.units_col, axis=1).columns
        data[cols] = self.normalize_col(data[cols], method)
        return data

    def construct_pre_int_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Construct pre intervention aggregate data to find synthetic control weights.

        Arguments:
            data (pd.DataFrame): Data frame of time series data from which to calculate the pre
            intervention features.

        Return:
            pre_int_data (pd.DataFrame): Data frame of pre intervention features for both test units
            and control units.
        """
        logging.debug("START construct_pre_int_data")

        # Break out pre intervention data from the time series data
        pre_int_data = data.query(f"{self.date_col} <= '{self.event_date}'")
        time_series_data = pre_int_data.copy()

        pre_int_data = pre_int_data.groupby(self.units_col, as_index=False).sum(
            numeric_only=True
        )
        pre_int_data = self.apply_aggregate_fn_map(pre_int_data)

        # Subset to the columns of interest before adding rolling means, or outcomes for each period
        pre_int_data = pre_int_data[self.out_cols]

        # In prospective modeling, rolling means are added
        if self.model_type == "prospective":
            rolling_means = self.calculate_rolling_means(time_series_data)
            pre_int_data = pre_int_data.merge(rolling_means, on=self.units_col)

        # In retrospective modeling, pre intervention outcomes are added for each time period
        if self.model_type == "retrospective":
            pre_int_data = self.add_outcomes(pre_int_data, time_series_data)

        pre_int_data = self.normalize_all(pre_int_data, method="standardization")

        logging.debug("END construct_pre_int_data")
        return pre_int_data

    def construct_numpy_arrays(
        self, pre_int_data: pd.DataFrame, test_units: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Return test unit vectors and feature matrix for control units along with two mappings.

        Donor map and factor map are created to ensure proper matching of the resulting synthetic
        control weights, and factor weights, when transitioning back to pandas data frames from
        numpy arrays.

        Arguments:
            pre_int_data (pd.DataFrame): The pre intervention data from which to construct the test
            unit vectors and control unit vectors/control unit feature matrix.

            test_units (str): List of test units to be used to construct the test unit vectors.

        Return:
            treatment_vectors (list): List of feature vectors for the provided test units.

            cntrl_feat_mtx (np.ndarray): Feature matrix containing a vector of features for each
            control unit.

            donor_map (pd.DataFrame): Data frame with a single column, self.units_col. The index
            stores the mapping.

            factor_map (pd.DataFrame): Data frame with a single column, "factor". The index stores
            the mapping.
        """
        logging.debug("START construct_numpy_arrays")

        # Treatment vectors
        treatment_vectors = [
            pre_int_data[pre_int_data[self.units_col] == test_unit]
            .drop(self.units_col, axis=1)
            .to_numpy()
            .T
            for test_unit in test_units
        ]

        cntrl_feat_mtx_df = self.exclude_test_units(pre_int_data, test_units)

        # Control unit feature matrix
        cntrl_feat_mtx = cntrl_feat_mtx_df.drop(self.units_col, axis=1).to_numpy().T

        # Donor pool map. Stores control units as a column, index acts as a mapping for the position
        # of the control unit in the cntrl_feat_mtx matrix columns.
        donor_map = cntrl_feat_mtx_df.reset_index(drop=True)[[self.units_col]]

        # Factor map. Stores factors as a column, index acts as a mapping for the position of the
        # factor in the cntrl_feat_mtx matrix rows.
        factor_map = cntrl_feat_mtx_df.drop(self.units_col, axis=1).T.reset_index()[
            ["index"]
        ]
        factor_map.columns = ["factor"]

        logging.debug("END construct_numpy_arrays")
        return treatment_vectors, cntrl_feat_mtx, donor_map, factor_map

    def minimize_cost_w(
        self,
        cost_w: Callable[[np.ndarray], float],
        n_donors: int,
        main: bool = False,
        test_unit: Union[str, None] = None,
    ) -> np.ndarray:
        """Minimize the given cost function using scipy's minimize tool.

        This method will find the optimal weights, i.e. synthetic control, under the given norm.

        Arguments:
            cost_w (Callable[[np.ndarray], float]): Cost function, calculates the cost for the given
            set of weights.

            n_donors (int): The number of control units.

            main (bool) = False: Flag for indicating if this is the main minimization run to find
            weights for the test units.

            test_unit (Union[str, None]) = None: For logging purposes only. Used to log which test
            unit this method is currently finding weights for.

        Return:
            w (np.ndarray): Vector of weights that will create the synthetic control.
        """
        logging.debug("START minimize_cost_w")
        minimize_methods = ["slsqp", "trust-constr"]
        slsqp_attempt_limit = 100
        max_attempts_message = (
            f"minimize_cost_w reached the maximum number of attempts: {self.max_attempts}. Increase"
            "max_attempts to increase the number of attempts."
        )
        # Bounds and constraints for the scipy minimize method
        # Must be non negative
        bnds = tuple([(0, None)] * n_donors)
        # The weights must sum to 1
        cons = {"type": "eq", "fun": lambda x: 1 - x.sum()}

        if main:
            logging.info(
                colors.yellow(
                    f"Starting minimization of cost function for {test_unit}."
                )
            )
            success = False
            failed = False
            attempts = 0

            # init the minimize_metadata_dict to track attempts and failed attempts
            minimize_metadata = dict()
            for method in minimize_methods:
                minimize_metadata[method] = dict()
                for count in ["attempts", "scipy_fail_count", "return_guess_count"]:
                    minimize_metadata[method][count] = 0

            while not (success or failed):
                if attempts < slsqp_attempt_limit:
                    minimize_method = "slsqp"
                    w0 = self.random_guess(n_donors)

                    if attempts == 0:
                        logging.info("Attempting minimization algo: slsqp")

                else:
                    minimize_method = "trust-constr"
                    # Need.ravel here because initial guess for 'trust-constr' must be one dimension
                    w0 = self.random_guess(n_donors).ravel()

                    if attempts == slsqp_attempt_limit:
                        logging.info("Attempting minimization algo: trust-constr")
                        # Constraints for trust-constr must be of type 'ineq'
                        cons = {"type": "ineq", "fun": lambda x: 1 - x.sum()}

                # Run scipy's minimize method and log attempts
                result = minimize(
                    cost_w, w0, bounds=bnds, constraints=cons, method=minimize_method
                )
                attempts += 1
                minimize_metadata[minimize_method]["attempts"] += 1
                w = result.x

                # Assess success conditions
                # Check if scipy logs are successful
                scipy_success = result.success
                # Check if scipy returned the initial guess
                returned_initial_guess = all(w0.flatten() == w)
                # Success criteria
                success = scipy_success and (not returned_initial_guess)

                # Check max attempts
                if (not success) and (attempts == self.max_attempts):
                    logging.info(max_attempts_message)
                    failed = True

                # Track failed attempts in minimize_metadata dict
                if scipy_success and returned_initial_guess:
                    minimize_metadata[minimize_method]["return_guess_count"] += 1
                elif not scipy_success:
                    minimize_metadata[minimize_method]["scipy_fail_count"] += 1

            # Create failed attempts summary for logging
            faild_attempts_summary = ""
            for method in minimize_methods:
                if minimize_metadata[method]["attempts"] > 0:
                    scipy_fail_count = minimize_metadata[method]["scipy_fail_count"]
                    return_guess_count = minimize_metadata[method]["return_guess_count"]
                    if scipy_fail_count + return_guess_count > 0:
                        faild_attempts_summary += (
                            f"Failed attempts summary for {method}:\n"
                        )
                        if scipy_fail_count > 0:
                            faild_attempts_summary += f"Scipy failed to minimize (count): {scipy_fail_count}\n"
                        if return_guess_count > 0:
                            faild_attempts_summary += f"Scipy returned the initial guess (count): {return_guess_count}\n"

            # Success message for logging. Include any failed attempts here.
            if success:
                msg = (
                    f"Scipy was able to successfully find weights for {test_unit} using "
                    f"{minimize_method} after {minimize_metadata[minimize_method]['attempts']} "
                    "attempt(s).\n" + faild_attempts_summary
                )
                logging.info(colors.green(msg))

            # Failure message for logging. Generate optimized random guess in the case of failure.
            else:
                msg = (
                    f"Scipy FAILED to find weights for {test_unit}!\n"
                    + faild_attempts_summary
                )
                logging.info(colors.red(msg))

                logging.info("Using optimized random guess instead.")
                # TODO: There a number of ways we can get a "good" weighting via brute force
                # and some strategic guessing. Come back to this and add in an optimized guess
                # method. For now take the best guess out of 1000.
                guesses = [self.random_guess(n_donors) for x in range(1000)]
                costs = np.array([cost_w(guess) for guess in guesses])
                w = guesses[costs.argmin()]

        # If not running as the main minimization task for the test units, a less rigorous approach
        # is preferred to cut down on run time and logging is unnecessary.
        else:
            w0 = self.random_guess(n_donors)
            result = minimize(cost_w, w0, bounds=bnds, constraints=cons, method="slsqp")
            w = result.x

        logging.debug("END minimize_cost_w")
        return w

    def find_lambda_upper_bound(
        self,
        treatment_vector: np.ndarray,
        cntrl_feat_mtx: np.ndarray,
        norm: Callable[[np.ndarray], float],
        placebo: bool,
    ) -> int:
        """Find the smallest lambda that results in a nearest neighbor match.

        As lambda ranges from 0 to inf, the resulting synthetic control will go from closest match
        in aggregate, to the nearest neighbor match. i.e. The larger lambda gets the more sparse the
        synthetic control weights get. Increase lambda by 1 until we have a single control unit with
        weight 1.

        Arguments:
            treatment_vector (np.ndarray): Feature vector for the test unit for which this method
            calculates an upper bound for.

            cntrl_feat_mtx (np.ndarray): Feature matrix for the control units.

        Return:
            lam (int): Upper bound for lambda.
        """
        logging.debug("START find_lambda_upper_bound")

        found_upper_bound = False
        found_max_lambda = False
        lam = 0
        n_donors = cntrl_feat_mtx.shape[1]

        # This number is based on how many decimal points would be required if each control unit
        # was weighted equally and 3 significant figures were needed e.g. With 527 control units
        # weighted equally, the weights would be 1/521 ~ 0.00191939. Keeping 3 sig figs would
        # require rounding to 5 decimal places.
        round_n_digits = len(str(n_donors)) + 2

        self.log_info(placebo, "Looking for an upper bound on lambda.")
        while not (found_upper_bound or found_max_lambda):
            cost_w = self.cost_w_factory(
                treatment_vector, cntrl_feat_mtx, norm=norm, lam=lam
            )

            w = self.minimize_cost_w(cost_w, n_donors)

            # An upper bound is found if all control units have weight 0 except for one.
            found_upper_bound = (np.round(w, round_n_digits) == 0).sum() == n_donors - 1

            if found_upper_bound:
                self.log_info(placebo, f"Found upper bound for lambda: {lam}")

            # Stop once lambda has been increased to the max_lambda
            found_max_lambda = lam == self.max_lambda
            if found_max_lambda:
                msg = f"Max lambda of {self.max_lambda} will be used for the lambda upper bound."
                self.log_info(placebo, msg)

            # If we have not found the upper bound or hit the max lambda. Increase lambda by 1 and
            # try again.
            if not (found_upper_bound or found_max_lambda):
                lam += 1

        logging.debug("END find_lambda_upper_bound")
        return lam

    def cost_w_factory(
        self,
        treatment_vector: np.ndarray,
        cntrl_feat_mtx: np.ndarray,
        norm: Callable[[np.ndarray], float],
        lam: float,
    ) -> Callable[[np.ndarray], float]:
        """Return a cost function for calculating the cost of a given synthetic control.

        The cost function is constructed using the provided norm and lambda. The cost is calculated
        as the norm of the difference of the treatment vector and the synthetic control, plus a
        penalty term weighted by lambda. This penalty term calculates for each control unit, the
        norm squared of the difference between the test unit vector, and the control unit vector,
        multiplied by the corresponding synthetic control weight for the given control unit.

        cost = norm(X - X_0 * W) + lam * SUM(w_j * ||X - X_j|| ** 2)

        norm: The norm specified by norm_type, euclidean or induced
        X   : Feature vector for the test uni
        X_0 : Feature matrix for all control units
        X_j : Feature vector for control unit 'j'
        W   : Synthetic control weights
        w_j : Synthetic control weight for control unit 'j'
        lam : Weight of the penalty term

        Arguments:
            treatment_vector (np.ndarray): Feature vector for a test unit.

            cntrl_feat_mtx (np.ndarray): Feature matrix for the control units.

            lam (float): Lambda. Weight of the penalty term.

        Return:
            cost_w (Callable[[np.ndarray], float]):
        """

        def cost_w(w: np.ndarray) -> float:
            return rust.cost_w(w, treatment_vector, cntrl_feat_mtx, lam)

        return cost_w

    def train_lambdas(
        self,
        data: pd.DataFrame,
        test_units: list,
        lambda_upper_bounds: List[float],
        placebo: bool,
    ) -> list:
        """Train the penalty weight, lambda, using one of two methods depending on the model type.

        For prospective modeling, split the pre intervention data into a training set and validation
        set. For retrospective modeling, use leave-one-out cross-validation on the pre and post
        intervention data for the control units.

        These techniques were motivated by Abadie and L'Hour and their paper "A Penalized Synthetic
        Control Estimator for Disaggregated Data."
        Reference: https://file.lianxh.cn/PDFTW/01621459.2021.pdf

        Arguments:
            data (pd.DataFrame): Time series data to use for training.

            test_units (list): List of test units to train lambdas for.

            lambda_upper_bounds (list): List of upper bounds to use for the training. One for each
            test unit.

        Return:
            lambdas (list): List of lambdas to use, one for each test unit.
        """
        logging.debug("START train_lambdas")

        if self.model_type == "prospective":
            msg = (
                "Training lambda using the train test split method on the pre intervention "
                "outcomes for the test unit."
            )
            self.log_info(placebo, msg)

            lambdas = self.train_test_split(
                data=data,
                test_units=test_units,
                lambda_upper_bounds=lambda_upper_bounds,
                placebo=placebo,
            )

        elif self.model_type == "retrospective":
            msg = "Training lambda using the cross validation method on the donor pool."
            self.log_info(placebo, msg)
            lambdas = self.cross_validate_controls(
                lambda_upper_bounds=lambda_upper_bounds, placebo=placebo
            )

        logging.debug("END train_lambdas")
        return lambdas

    def cross_validate_controls(
        self, lambda_upper_bounds: List[float], placebo: bool
    ) -> float:
        """Train lambda using the post intervention outcomes for the control units.

        Leave-one-out cross-validation on the control units using pre and post intervention data.
        This is done without data from the test units. Take training data as pre intervention data.
        Use post intervention outcomes as the validation data. The cost is taken as the sum of the
        squared prediction error for post intervention data over all control units. This results
        in a single lambda. Then min(lambda, lambda_upper_bound) is taken as the resulting lambda
        for each lambda_upper_bound in lambda_upper_bounds list.

        Arguments:
            data (pd.DataFrame): Time series data. The test units will be removed in this step.

            lambda_upper_bounds (list): List of upper bounds for lambda, one for each test unit.

        Return:
            lambdas (list): List of optimal lambdas, one for each test unit.
        """
        # In case the find_weights method is called multiple times externally, need to reset this
        # attribute to None so that it can be properly retrained.
        self.cross_validated_lambda = None
        cost_lambda = self.cost_lambda_retrospective_factory()

        # Bounds for the minimize_scalar method. Must be non negative and bounded by the lambda
        # upper bound to increase speed.
        bnds = (0, max(lambda_upper_bounds))

        result = minimize_scalar(cost_lambda, bounds=bnds, method="bounded")
        if not result.success:
            self.log_info(placebo, "Scipy failed to find an optimal lambda.")

        lam = result.x

        # Clip the resulting lambda using the upper bounds.
        lambdas = [min(lam, upper_bound) for upper_bound in lambda_upper_bounds]

        # Save this lambda so it can be reused during the placebo test to cut down on time
        # dramatically. This is technically not perfect but very close to it and is well worth
        # the time saved.
        self.cross_validated_lambda = lam

        return lambdas

    def grid_search(
        self,
        cost_lambda: Callable[[float], float],
        current_round: int,
        lambda_upper_bound: int,
        test_lambdas: List[float],
    ) -> Tuple[float, List[float]]:
        """Search test_lambdas for the lambda with the smallest cost.

        Helper function used in the train_test_split method.

        Arguments:
            cost_lambda (Callable[[float], float]): Calculate the cost of a given lambda.

            current_round (int): Indicates which round of the grid search is in progress.

            lambda_upper_bound (int): Upper bound for lambda.

            test_lambdas (List[float]): List of lambdas to search through.

        Return:
            arg_min_lam (float): The lambda from test_lambdas with the lowest cost.

            new_test_lambdas (List[float]): List of lambdas to search next.
        """
        # Find the lambda with the smallest cost
        errors = np.array([cost_lambda(lam) for lam in test_lambdas])
        arg_min_lam = test_lambdas[errors.argmin()]
        logging.debug(f"Round {current_round} best lambda: {arg_min_lam}")
        logging.debug(f"With error: {errors.min()}")

        # Create a new grid to search by zooming in around the current optimal lambda.
        radius = 0.9 / (10**current_round)
        a = max(arg_min_lam - radius, 0)
        b = min(arg_min_lam + radius, lambda_upper_bound)
        new_test_lambdas = np.linspace(a, b, 19)
        return arg_min_lam, new_test_lambdas, errors.min()

    def train_test_split(
        self,
        data: pd.DataFrame,
        test_units: list,
        lambda_upper_bounds: List[float],
        placebo: bool,
    ) -> float:
        """Train lambda using the pre intervention outcomes for the treated unit.

        For each test unit in test_units. Take the time series data for the pre intervention
        outcomes and split it into two sets of time series data, training and validation. Take any
        features and aggregates from the training data to train a synthetic control, then calculate
        the MSPE on the validation set as the cost. The cost is minimized by using a brute force
        grid search. Fortunately, the upper bound on lambda makes this grid search effective. For a
        given upper bound on lambda, test all ints from 0 to lambda_upper_bound. Take the lambda
        with the minimum cost and create a new search of the space argm_min +/- 0.9 with 10
        intervals. Repeat this again with arg_min +/- 0.09 with 10 intervals to get the final
        arg_min_lambda.

        Arguments:
            data (pd.DataFrame): Time series data.

            test_units (list): List of test units to train lambdas for.

            lambda_upper_bounds (list): List of upper bounds for lambda for each test unit.

        Return:
            lambdas (list): List of optimal lambdas, one for each test unit.
        """
        logging.debug("START train_test_split")

        (
            treatment_vectors,
            cntrl_feat_mtx,
            donor_map,
            _,
            test_data,
        ) = self.construct_train_test_split_data(data, test_units)

        norm = self.norm_factory()

        self.log_info(placebo, "Generating cost functions for lambdas")
        lambda_cost_fns = [
            self.cost_lambda_prospective_factory(
                test_unit=test_unit,
                treatment_vector=treatment_vector,
                cntrl_feat_mtx=cntrl_feat_mtx,
                test_data=test_data,
                donor_map=donor_map,
                norm=norm,
            )
            for test_unit, treatment_vector in zip(test_units, treatment_vectors)
        ]

        # This is a brute force grid search approach for minimizing lambda cost functions
        lambdas = []
        for cost_lambda, lambda_upper_bound in zip(
            lambda_cost_fns, lambda_upper_bounds
        ):
            # Start by testing every integer value lambda from 0 to lambda_upper_bound
            test_lambdas = list(range(lambda_upper_bound + 1))
            for current_round in range(4):
                # The test_lambdas are updated by gird_search each iteration and fed back into
                # grid_search
                arg_min_lam, test_lambdas, min_error = self.grid_search(
                    cost_lambda, current_round, lambda_upper_bound, test_lambdas
                )

            self.log_info(placebo, f"Best overall lambda: {arg_min_lam}")
            self.log_info(placebo, f"With error: {min_error}")

            lambdas.append(arg_min_lam)

        logging.debug("END train_test_split")
        return lambdas

    def construct_train_test_split_data(
        self, data: pd.DataFrame, test_units: List[str]
    ) -> Tuple[List[np.ndarray], np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return all objects required for the train test split process.

        Arguments:
            data (pd.DataFrame): Time series data to be used for train test split.

        Return:
            treatment_vectors (List[np.ndarray]): list of treatment vectors.

            cntrl_feat_mtx (np.ndarray): Feature matrix for control units.

            donor_map (pd.DataFrame): Data frame with a single column, self.units_col. The index
            stores the mapping.

            factor_map (pd.DataFrame): Data frame with a single column, "factor". The index stores
            the mapping.

            test_data (pd.DataFrame): Data frame of time series data that will be used to test the
            resulting synthetic control from the training data.
        """
        logging.debug("START construct_train_test_split_data")

        # Create training data and save a copy to calculate rolling means from it later
        training_data = data.query(f"{self.date_col} <= '{self.train_test_split_date}'")
        time_series_data = training_data.copy()

        # Aggregate data to get features for training
        training_data = training_data.groupby(self.units_col, as_index=False).sum(
            numeric_only=True
        )
        training_data = self.apply_aggregate_fn_map(training_data)
        training_data = training_data[self.out_cols]

        # Calculate rolling means and add them to training_data as additional features
        rolling_means = self.calculate_rolling_means(time_series_data)
        training_data = training_data.merge(rolling_means, on=self.units_col)
        training_data = self.normalize_all(training_data, method="standardization")

        # Create the validation time series data. This will be used to calculate cost as prediction
        # error.
        test_data = data.query(f"{self.date_col} <= '{self.event_date}'").query(
            f"{self.date_col} > '{self.train_test_split_date}'"
        )

        # Create numpy arrays, a feature vector for each test unit, and a feature matrix for the
        # control units.
        (
            treatment_vectors,
            cntrl_feat_mtx,
            donor_map,
            factor_map,
        ) = self.construct_numpy_arrays(training_data, test_units)

        logging.debug("END construct_train_test_split_data")
        return treatment_vectors, cntrl_feat_mtx, donor_map, factor_map, test_data

    def cost_lambda_retrospective_factory(self) -> Callable[[float], float]:
        """Return cost function for lambda using retrospective modeling.

        Return:
            cost_lambda (Callable[[int], float]): Cost function for lambda.
        """
        logging.debug("START cost_lambda_retrospective")

        def cost_lambda(lam: float) -> float:
            """Fit a synthetic control for each control unit and aggregate the prediction error.

            The pre intervention data is used to train the weights, the resulting synthetic control
            is then fit to the post intervention data. The prediction error is measured on the fit
            of the post intervention data. This prediction error is aggregated across all control
            units and returned as the cost of lambda.

            Arguments:
                lam (float): Lambda, weight of the penalty term.

            Return:
                cost (float): Sum of squared prediction errors.
            """
            data = self.exclude_test_units(self.data, self.test_units)

            # Fit synthetic controls.
            control_diffs = []
            for control in self.control_units:
                sc_diff = self.fit_control_unit(control, data, lam)
                control_diffs.append(sc_diff)

            # Concat the diffs for all the control units and subset to post intervention data.
            df_control_diffs = pd.concat(control_diffs).query(
                f"{self.date_col} > '{self.event_date}'"
            )

            # Aggregate prediction error as cost.
            cost = (
                df_control_diffs.groupby(self.units_col).agg({"diff": "sum"})["diff"]
                ** 2
            ).sum()

            return cost

        logging.debug("END cost_lambda_retrospective")
        return cost_lambda

    def cost_lambda_prospective_factory(
        self,
        test_unit: str,
        treatment_vector: np.ndarray,
        cntrl_feat_mtx: np.ndarray,
        test_data: pd.DataFrame,
        donor_map: pd.DataFrame,
        norm: Callable[[np.ndarray], float],
    ):
        """Construct cost function for lambda using prospective modeling.

        Arguments:
            test_unit (str)

            treatmnet_vector (np.ndarray): Feature vector for the given test unit.

            cntrl_feat_mtx (np.ndarray): Feature matrix for the control units.

            test_data (pd.DataFrame): Data to construct a counterfactual/synthetic control for.

            donor_map (pd.DataFrame): Mapping of control units to their index in the synthetic
            control weight vector.

        Return:
            cost_lambda (Callable[[float], float])
        """

        def cost_lambda(lam: float) -> float:
            """Calculate the cost of the given lambda.

            Calculate the cost of the given lambda by constructing synthetic control data using
            lambda in the cost function for w then measuring the prediction error.

            Arguments:
                lam (float): Lambda, weight of the penalty term in cost_w.

            Return:
                mspe (float): Mean squared prediction error.
            """
            # Generate cost function for a set of weights w.
            cost_w = self.cost_w_factory(
                treatment_vector=treatment_vector,
                cntrl_feat_mtx=cntrl_feat_mtx,
                norm=norm,
                lam=lam,
            )

            # Minimize cost_w
            n_donors = cntrl_feat_mtx.shape[1]
            w = self.minimize_cost_w(cost_w, n_donors)

            # Fit synthetic control data using the resulting weights w
            _, synthetic_control = self.construct_synthetic_ts_data(
                w=w,
                donor_map=donor_map,
                time_series_data=test_data,
                test_unit=test_unit,
            )

            # Calculate the prediction error
            mspe = calculate_mspe(
                df=synthetic_control, col1=f"synthetic_{test_unit}", col2=test_unit
            )

            return mspe

        return cost_lambda

    def construct_synthetic_ts_data(
        self,
        w: np.ndarray,
        donor_map: pd.DataFrame,
        time_series_data: pd.DataFrame,
        test_unit: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Construct synthetic control time series data with the given set of weights w.

        Return two data frames, one long, one wide. Both contain time series data for the given
        test unit and its corresponding synthetic control.

        Arguments:
            w (np.ndarray): Weights to be used with the control units to construct the
            counterfactual/synthetic control.

            donor_map (pd.DataFrame): Mapping of control units to their index in the synthetic
            control weight vector.

            time_series_data (pd.DataFrame): Time series data used to construct the counterfactual.

            test_unit (str): Test unit for which to construct the counterfactual.

        Return:
            synthetic_control_ts (pd.DataFrame): Time series data of the counterfactual for the
            provided test unit.

            sc_wide (pd.DataFrame): The synthetic_control_ts data frame converted from long data to
            wide.
        """
        logging.debug("START construct_synthetic_ts_data")

        # Map the vector of weights w onto the corresponding control units
        df_donor_weights = self.join_weights_to_map(w, donor_map)

        # Exclude test unit data from the time series data
        if test_unit not in self.test_units:
            # In the case this is being used for a placebo test, add the current control unit
            test_units = self.test_units + [test_unit]
        else:
            test_units = self.test_units

        synthetic_control_ts = self.exclude_test_units(time_series_data, test_units)

        # Join on the synthetic control weights to the time series data
        synthetic_control_ts = synthetic_control_ts.merge(
            df_donor_weights, on=self.units_col
        )

        # Take the product of the outcome metric and the weight column
        synthetic_control_ts[f"weighted_{self.outcome_metric}"] = (
            synthetic_control_ts[self.outcome_metric] * synthetic_control_ts["weight"]
        )

        # Group by date and sum the weighted outcome metric to achieve the weighted average that
        # makes up the synthetic control data
        synthetic_control_ts = (
            synthetic_control_ts.groupby(self.date_col, as_index=False)
            .agg({f"weighted_{self.outcome_metric}": "sum"})
            .rename(columns={f"weighted_{self.outcome_metric}": self.outcome_metric})
        )

        # Add column for units and fill with f"synthetic_{test_unit}"
        synthetic_control_ts[self.units_col] = f"synthetic_{test_unit}"
        synthetic_control_ts = synthetic_control_ts[
            [self.date_col, self.units_col, self.outcome_metric]
        ]

        # Query the time series data for the test unit
        test_unit_ts = time_series_data.query(f"{self.units_col} == '{test_unit}'")[
            [self.date_col, self.units_col, self.outcome_metric]
        ]

        # Append the test unit time series data to the synthetic control time series data
        synthetic_control_ts = pd.concat([synthetic_control_ts, test_unit_ts])

        # Construct wide version of data to analyze the diff
        sc_wide = synthetic_control_ts.pivot_table(
            index=self.date_col, columns=self.units_col, values=self.outcome_metric
        ).reset_index()
        sc_wide.columns = sc_wide.columns.tolist()

        # Calculate the diff
        sc_wide["diff"] = sc_wide[test_unit] - sc_wide[f"synthetic_{test_unit}"]

        synthetic_control_ts = synthetic_control_ts.reset_index(drop=True)
        sc_wide = sc_wide.reset_index(drop=True)

        logging.debug("END construct_synthetic_ts_data")
        return synthetic_control_ts, sc_wide

    def find_weights(
        self,
        data: Union[pd.DataFrame, None] = None,
        test_units: Union[List[str], None] = None,
        lam: Union[float, None] = None,
        placebo: bool = False,
    ):
        """Find synthetic control weights for each test unit.

        Weights are optimized by minimizing the cost function below. This is the main method for the
        synthetic control algorithm. Moreover, this method is used internally for the placebo test
        which is indicated by the placebo argument. When placebo is set to True, logging is turned
        off via the self.log_info wrapper. When this method is called externally it is intended to
        be called with no arguments, e.g. sc.find_weights(), however if it is called internally, we
        need to pass some information through.

        norm(X_i - X_0 * W) + lam * SUM(w_j * ||X_i - X_j|| ** 2)
        Where w_j >= 0 and SUM(w_j over j) = 1

        norm: The norm specified by norm_type, euclidean or induced
        X_i : Feature vector for test unit 'i'
        X_0 : Feature matrix for all control units
        X_j : Feature vector for control unit 'j'
        W   : Synthetic control weights
        w_j : Synthetic control weight for control unit 'j'
        lam : Weight of the penalty term

        Lambda is the weight of the penalty term and is trained prospectively or retrospectively.
        The prospective model trains lambda by splitting the pre intervention data for the treated
        unit into training and test data sets. The retrospective model trains lambda using the
        control units and cross validation on the pre and post intervention data.

        Arguments:
            data (Union[pd.DataFrame, None]) = None: Data to be used for finding synthetic control
            weights. If None, defaults to self.data.

            test_units (Union[List[str], None]) = None: Test units to find synthetic control
            weights for. If None, defaults to self.test_units.

            lam (Union[float, None]) = None: Weight for the penalty term in the synthetic control
            weights cost function. If None, defaults to self.lam. If self.lam is also none then
            lambda(s) will be trained.

            placebo (bool) = False: Indicates if find_weights is being called for a control unit.
            This happens for lambda cost functions and the placebo test.

        Results:
            self.weights (List[np.ndarray]): List of weight vectors, one for each test unit.

            self.weights_dfs (List[pd.DataFrame]): List of weight vectors represented as data frames
        """
        data, test_units, lam = self.getargs(locals(), ["data", "test_units", "lam"])

        if (not placebo) and (self.norm_type == "induced"):
            self.factor_weights = self.train_factor_weights(data, test_units)

        pre_int_data = self.construct_pre_int_data(data)
        (
            treatment_vectors,
            cntrl_feat_mtx,
            donor_map,
            factor_map,
        ) = self.construct_numpy_arrays(pre_int_data, test_units)

        norm = self.norm_factory()

        # Get values for lambda. Lambda is not None in three cases. (1) If lambda is provided as an
        # instance attribute upon instantiation. (2) When calculating the cost of a lambda during
        # training. (3) When running the placebo test.
        if lam is not None:
            self.log_info(placebo, f"Using hard coded lambda: {lam}")

            # Check if the model is retrospective
            retrospective = self.model_type == "retrospective"
            # Check if a lambda has been trained via cross validation
            cross_validated_lambda = (
                getattr(self, "cross_validated_lambda", None) is not None
            )

            # These three conditions need to be checked due to a few edge cases that occur when the
            # find weights method is called multiple times externally.
            if placebo and retrospective and cross_validated_lambda:
                # Calculate lambda upper bounds
                lambda_upper_bounds = [
                    self.find_lambda_upper_bound(
                        treatment_vector=treatment_vector,
                        cntrl_feat_mtx=cntrl_feat_mtx,
                        norm=norm,
                        placebo=placebo,
                    )
                    for treatment_vector in treatment_vectors
                ]
                # This is required in case a cross validated lambda is being used in the placebo
                # test. If the lambda upper bound happens to be smaller than the cross validated
                # lambda, it needs to be trimmed. In this case, find_weights is only being run
                # on a single unit and the list comprehension is redundant, but, future proof.
                lambdas = [min(lam, upper_bound) for upper_bound in lambda_upper_bounds]

            else:
                lambdas = [lam] * len(test_units)

            cost_w_fns = [
                self.cost_w_factory(treatment_vector, cntrl_feat_mtx, norm, lam)
                for treatment_vector in treatment_vectors
            ]

        # If no lambda is provided, train a lambda for each test unit.
        else:
            n_test_units = len(treatment_vectors)
            self.log_info(placebo, f"Training {n_test_units} lambda(s)")

            # Calculate lambda upper bounds
            lambda_upper_bounds = [
                self.find_lambda_upper_bound(
                    treatment_vector, cntrl_feat_mtx, norm, placebo
                )
                for treatment_vector in treatment_vectors
            ]

            # Train lambdas
            lambdas = self.train_lambdas(
                data=data,
                test_units=test_units,
                lambda_upper_bounds=lambda_upper_bounds,
                placebo=placebo,
            )

            # Generate cost functions for each test unit
            cost_w_fns = [
                self.cost_w_factory(
                    treatment_vector, cntrl_feat_mtx, norm=norm, lam=lam
                )
                for treatment_vector, lam in zip(treatment_vectors, lambdas)
            ]

        self.log_info(placebo, f"Using lambdas: {lambdas}")

        # Solve each minimization problem for each test unit
        n_donors = cntrl_feat_mtx.shape[1]
        weights = {
            test_unit: self.minimize_cost_w(cost_w, n_donors, not placebo, test_unit)
            for test_unit, cost_w in zip(test_units, cost_w_fns)
        }
        weights_dfs = {
            test_unit: self.join_weights_to_map(weights[test_unit], donor_map)
            for test_unit in test_units
        }

        # If find weights is being called for a placebo run, return the results.
        if placebo:
            return donor_map, weights, weights_dfs

        # Otherwise set the results as instance attributes.
        else:
            if getattr(self, "factor_weights") is not None:
                self.factor_weights_dict = (
                    self.join_weights_to_map(self.factor_weights, factor_map)
                    .set_index("factor")["weight"]
                    .to_dict()
                )
            else:
                self.factor_weights_dict = "NA"

            self.donor_map = donor_map
            self.lambdas = {
                test_unit: lam for test_unit, lam in zip(test_units, lambdas)
            }
            self.pre_int_data = pre_int_data
            self.weights = weights
            self.weights_dfs = weights_dfs

    def fit_synthetic_control(
        self,
        data: Union[pd.DataFrame, None] = None,
        test_units: Union[List[str], None] = None,
        donor_map: Union[pd.DataFrame, None] = None,
        weights: Union[Dict[str, np.ndarray], None] = None,
        placebo: bool = False,
    ):
        """Construct counterfactual time series data for each test unit.

        Iterates over the construct_synthetic_ts_data method. This method should be called after the
        find weights method. Intended to be called with no arguments when used externally, e.g.
        sc.fit_synthetic_control(). However, this method is used internally as well, in which case
        we need to pass some information through.

        Arguments:
            data (Union[pd.DataFrame, None]) = None: Time series data to be used with the provided
            weights to construct counterfactual/synthetic control.

            test_units (Union[List[str]) = None: Test units to construct counterfactual for.

            donor_map (Union[pd.DataFrame, None]) = None: Maps vector of weights to their
            corresponding control units.

            weights (Union[List[np.ndarray], None]) = None: List of weight vectors for computing
            counterfactual.

            placebo (bool) = False: Indicates if find_weights is being called for a control unit.
            This happens for lambda cost functions and the placebo test.

        Results:
            self.sc_fits (list): List of tuples for each test unit. Each tuple contains two data
            frames, one long, and one wide, containing time series data for the counterfactual.
        """
        args = ["data", "test_units", "donor_map", "weights"]
        data, test_units, donor_map, weights = self.getargs(locals(), args)

        msg = (
            "Cannot fit synthetic control data without weights. Train synthetic control weights "
            "first with the find_weights method."
        )
        assert weights is not None, msg

        sc_fits = {
            test_unit: self.construct_synthetic_ts_data(
                w=weights[test_unit],
                donor_map=donor_map,
                time_series_data=data,
                test_unit=test_unit,
            )
            for test_unit in test_units
        }

        if placebo:
            return sc_fits
        else:
            self.sc_fits = sc_fits

    def calculate_fit_error(self):
        """Calculate pre and post rmspe for the test units."""
        sc_fits = getattr(self, "sc_fits", None)
        error_type = "rmspe"

        msg = "Cannot calculate fit error. Run the fit_synthetic_control method first."
        assert sc_fits is not None, msg

        pre_fit_errors = []
        post_fit_errors = []
        for test_unit in self.test_units:
            fit = self.sc_fits[test_unit]
            pre_fit_error = np.sqrt(
                (
                    fit[1].query(f"{self.date_col} <= '{self.event_date}'")["diff"] ** 2
                ).mean()
            )

            post_fit_error = np.sqrt(
                (
                    fit[1].query(f"{self.date_col} > '{self.event_date}'")["diff"] ** 2
                ).mean()
            )

            pre_fit_errors.append(pre_fit_error)
            post_fit_errors.append(post_fit_error)

        self.test_unit_prediction_error = pd.DataFrame(
            {
                "test_unit": self.test_units,
                "pre_fit_error": pre_fit_errors,
                "post_fit_error": post_fit_errors,
                "error_type": error_type,
            }
        )

    def estimate_effects(self) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Estimate the mean treatment effect across all test units.

        Return:
            mean_diff (pd.DataFrame): Data frame of time series data containing the mean diff
            between treatment units and their corresponding synthetic controls.

            mean_effects (pd.DataFrame): The mean_diff data frame above subset to the treatment
            period.

            mean_effects_agg (float): The mean of the time series data in mean_effects.
        """
        # Calculate mean diff between treatment units and their corresponding synthetic controls
        # over the entire available time period
        mean_diff = (
            pd.concat(
                [fit[1][[self.date_col, "diff"]] for fit in self.sc_fits.values()]
            )
            .groupby(self.date_col, as_index=False)
            .mean()
        )

        # Subset mean_diff to the treatment period
        mean_effects = mean_diff.query(f"{self.date_col} > '{self.event_date}'")

        # Calculate the mean of the daily treatment effects.
        mean_effects_agg = mean_effects["diff"].mean()

        return mean_diff, mean_effects, mean_effects_agg

    def train_factor_weights(
        self, data: pd.DataFrame = None, test_units: List[np.ndarray] = None
    ) -> np.ndarray:
        """Estimate the predictive power of the features used to construct the synthetic control.

        Predictive power is estimated by solving a separate, multiple regression, problem. Namely,
        minimize

        (1 / (2J))||Y - XV||Fro**2 + alpha||V||21

        Where X is the feature matrix for the control units and Y are the time series outcomes for
        each control unit for the outcome metric of interest. This is done with sklearn's
        MultiTaskLasso multiple regression model. The resulting matrix V has multiple weights for
        each factor. To reduce the dimensionality, we square each entry and sum along the axis of
        the factors. Then take the square root of each entry. Giving us a vector of factor weights.
        e.g.

        np.sqrt(np.sum(np.square(V), axis=0))

        Results:
            factor_weights (np.ndarray)
        """
        logging.debug("START train_factor_weights")

        data, test_units = self.getargs(locals(), ["data", "test_units"])

        if self.model_type == "prospective":
            _, training_data, _, _, _ = self.construct_train_test_split_data(
                data, test_units
            )

            validation_data = (
                self.exclude_test_units(data, test_units)
                .query(f"{self.date_col} <= '{self.event_date}'")
                .query(f"{self.date_col} > '{self.train_test_split_date}'")
            )

        elif self.model_type == "retrospective":
            pre_int_data = self.construct_pre_int_data(data)
            _, training_data, _, _ = self.construct_numpy_arrays(
                pre_int_data, test_units
            )

            validation_data = self.exclude_test_units(data, test_units).query(
                f"{self.date_col} > '{self.event_date}'"
            )

        training_data = training_data.T

        validation_data[self.outcome_metric] = self.standardize_col(
            validation_data[self.outcome_metric]
        )

        validation_data = validation_data.pivot_table(
            index=self.date_col, columns=self.units_col, values=self.outcome_metric
        ).values.T

        # GridSearch for best alpah
        mtl = MultiTaskLasso(max_iter=10000, tol=0.001)
        parameters = {"alpha": [1 / (10**i) for i in range(16)]}
        clf = GridSearchCV(mtl, parameters)
        clf.fit(training_data, validation_data)
        best_alpha = clf.best_params_["alpha"]

        # train model with the best parameter
        clf = MultiTaskLasso(alpha=best_alpha, max_iter=10000, tol=0.001)
        clf.fit(training_data, validation_data)
        # Reduce dimensionality of the results to get one weight per feature
        factor_weights = np.sqrt(np.sum(np.square(clf.coef_), axis=0))

        if (factor_weights == 0).all():
            factor_weights = np.array([1] * len(factor_weights))

        logging.debug("END train_factor_weights")
        return factor_weights

    def fit_control_unit(
        self, control_unit: str, data: pd.DataFrame, lam: Union[float, None]
    ) -> pd.DataFrame:
        """Find weights and fit synthetic control data to the given control unit.

        Arguments:
            control_unit (str): Control unit to get synthetic control for.

            data (pd.DataFrame): Data used to construct synthetic control. This should not include
            data from the test units.

            lam (Union[float, None]): Lambda to be used in the find weights method.

        Return:
            sc_diff (pd.DataFrame): Data frame of synthetic control data from the
            fit_synthetic_control method.
        """
        out_cols = [self.date_col, self.units_col, "diff"]
        test_units = [control_unit]

        # Find optimal weighting
        donor_map, weights, _ = self.find_weights(
            data=data, test_units=test_units, lam=lam, placebo=True
        )

        # Fit synthetic control data using weights
        sc_diff = self.fit_synthetic_control(
            data=data,
            test_units=test_units,
            donor_map=donor_map,
            weights=weights,
            placebo=True,
        )[control_unit][1].copy()

        # Construct data frame with diff for the control unit
        sc_diff[self.units_col] = control_unit
        sc_diff = sc_diff[out_cols]
        return sc_diff

    def run_placebo_test(self):
        """Run the synthetic control algorithm on all control units.

        Iterate fit_control_unit method over the control units. Results in the instance attribute
        self.rmspe_ratios. An rmspe_ratio is the ratio of the rmspe (root mean squared prediction
        error) for a synthetic control for the pre and post intervention fit. This is calculated for
        all control units and all test units.
        """
        logging.debug("START run_placebo_test")
        out_cols = [self.date_col, self.units_col, "diff"]
        data = self.exclude_test_units(self.data, self.test_units)
        error_type = "rmspe"
        conditions = {
            "separate": (
                "test unit rmspe_ratio is greater than all control unit rmspe_ratios"
            ),
            "far": (
                "test unit rmspe_ratio is at least 2 standard deviations away from the control "
                "unit rmspe_ratio mean"
            ),
            "pass": "test unit passes both separate and far tests.",
        }

        # If lambda has been trained via cross validation there is no need to train lambda again.
        lam = getattr(self, "cross_validated_lambda", None)

        control_diffs = []
        for control in self.control_units:
            sc_diff = self.fit_control_unit(control, data, lam)
            control_diffs.append(sc_diff)

        # Construct main data frame for control diffs
        df_control_diffs = pd.concat(control_diffs)

        # If a synthetic control has been constructed for test units, include it here, otherwise
        # only consider the diffs for the control units.
        test_unit_diffs = []
        if getattr(self, "sc_fits", None) is not None:
            for test_unit in self.test_units:
                sc_data = self.sc_fits[test_unit]
                df_diff = sc_data[1]
                df_diff[self.units_col] = test_unit
                test_unit_diffs.append(df_diff[out_cols])

            df_test_unit_diffs = pd.concat(test_unit_diffs)

            df_diffs = pd.concat([df_test_unit_diffs, df_control_diffs])

        else:
            df_diffs = df_control_diffs.copy()

        # Calculate the squared prediction error
        df_diffs["spe"] = df_diffs["diff"] ** 2

        # Construct pre and post rmspe data frames
        # Calculate the mean squared prediction error
        pre_fit_error = (
            df_diffs.query(f"{self.date_col} <= '{self.event_date}'")
            .groupby(self.units_col, as_index=False)
            .agg({"spe": "mean"})
            .rename(columns={"spe": "mspe"})
        )

        post_fit_error = (
            df_diffs.query(f"{self.date_col} > '{self.event_date}'")
            .groupby(self.units_col, as_index=False)
            .agg({"spe": "mean"})
            .rename(columns={"spe": "mspe"})
        )

        # Calculate the root mean squared prediction error
        pre_fit_error["pre_fit_error"] = np.sqrt(pre_fit_error["mspe"])
        post_fit_error["post_fit_error"] = np.sqrt(post_fit_error["mspe"])

        pre_fit_error = pre_fit_error[[self.units_col, "pre_fit_error"]]
        post_fit_error = post_fit_error[[self.units_col, "post_fit_error"]]

        # Calculate the rmspe ratios
        error_ratios = pre_fit_error.merge(
            post_fit_error, how="left", on=self.units_col
        )
        error_ratios["error_ratio"] = (
            error_ratios["post_fit_error"] / error_ratios["pre_fit_error"]
        )
        error_ratios["error_type"] = error_type

        # Only perform separate and far tests if a post intervention period exists.
        if self.post_int_exists:
            # Determine if the test unit rmspe ratio is separate and far from the control units
            control_error_ratios = self.exclude_test_units(
                error_ratios, self.test_units
            )["error_ratio"].to_numpy()

            # Standard deviation for the distribution of rmspe ratios for the control units
            control_mean = control_error_ratios.mean()
            control_std = np.std(control_error_ratios)

            def separate(row: pd.Series) -> bool:
                """Success criteria for being separate."""
                return (row["error_ratio"] > control_error_ratios).all()

            def far(row: pd.Series) -> bool:
                """Success criteria for being far."""
                return (row["error_ratio"] - control_mean) > 2 * control_std

            # Apply separate and far tests then evaluate which units pass the tests.
            error_ratios["error_ratio_mean"] = control_mean
            error_ratios["error_ratio_sd"] = control_std
            error_ratios["separate"] = error_ratios.apply(separate, axis=1)
            error_ratios["far"] = error_ratios.apply(far, axis=1)
            error_ratios["pass"] = error_ratios["separate"] & error_ratios["far"]
            error_ratios["unit_type"] = error_ratios[self.units_col].apply(
                lambda unit: "test" if unit in self.test_units else "control"
            )
            error_ratios["conditions"] = [conditions] * len(error_ratios)

        self.placebo_diffs = df_diffs[out_cols]
        self.error_ratios = error_ratios
        self.placebo_test_conditions = conditions

        logging.debug("END run_placebo_test")
