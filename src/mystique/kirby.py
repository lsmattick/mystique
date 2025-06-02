"""Class for Kirby algo."""
import colorful as colors
import logging
import pandas as pd
from typing import Callable, Union

from .base import MystiqueBase
from .utils import euclid_distance
from mystique import rust


class Kirby(MystiqueBase):
    """Selects optimal control units for the given test units from the control unit candidates.

    Kirby measures distance between a test unit and possible control units by measuring the
    distance between their corresponding time series data in the pre intervention period for the
    given outcome metric. The distance metric used is either the standard euclidean distance or the
    dynamic time warping distance metric (dtw). The closest n control units are picked for each
    control unit where n is specified by the instance attribute n_neighbors. If there are multiple
    test units it is possible and likely that these n control units will overlap.

    data: pd.DataFrame
        Pandas data frame containing the time series data for the outcome variable and any
        covariates to be used.

    test_units: list,
        List of units that received treatment.
        e.g. ["campaign1", "campaign2"]

    units_col: str,
        Column name from the data frame provided in data to look up test units and control units.

    outcome_metric: str,
        Column name from the data frame provided in data that contains the outcome metric of
        interest.

    date_col: str,
        Column name from the data frame provided in data that contains the date.

    event_date: Union[str, None] = None,
        Date "YYYY-MM-DD" that marks when the treatment started. This date will be included in the
        pre intervention data.

    distance_metric: Union[str, None] = None,
        Default to dtw. Name of the distance metric to use when measuring distance between two sets
        of time series data. This can be either dtw or euclidean.

    n_neighbors: Union[int, None] = None,
        The number of control units to select for each test unit. If this is left as None then Kirby
        will default to 19 control units if possible. If there are less control candidates than 19
        then Kirby will return all control units.
    """

    supported_distance_metrics = ["dtw", "euclidean"]

    default_distance_metric = "dtw"
    default_n_neighbors = 19

    def __init__(
        self,
        data: pd.DataFrame,
        test_units: list,
        units_col: str,
        outcome_metric: str,
        date_col: str,
        event_date: Union[str, None] = None,
        distance_metric: Union[str, None] = None,
        n_neighbors: Union[int, None] = None,
    ):
        self.data = data
        self.test_units = test_units
        self.units_col = units_col
        self.outcome_metric = outcome_metric
        self.event_date = event_date
        self.date_col = date_col
        self.distance_metric = distance_metric or self.default_distance_metric
        self.n_neighbors = n_neighbors

        self._validate_event_date_format()
        self._validate_columns()
        self._validate_distance_metric()

        self.data = self._preprocess_data(self.data, set_index=True)

        self._validate_test_units(index=True)
        self._validate_event_date_in_range()
        self._validate_data_pre_event_date()

        self.event_date = self._coalesce_event_date()
        self.units = self.data.index.unique().tolist()
        self.control_units = list(set(self.units) - set(self.test_units))
        self.test_unit_idxs = [self.units.index(unit) for unit in self.test_units]
        self.n_units = len(self.units)
        self.n_control_units = self.n_units - len(self.test_units)
        self.n_neighbors = self._coalesce_n_neighbors()

        self._validate_n_neighbors()

    def _validate_distance_metric(self):
        """Assert distance_metric is in the list of supported distance metrics."""
        msg = (
            "Distance metric provided is not supported. Please use one of the supported distance "
            f"metrics: {self.supported_distance_metrics}"
        )
        assert self.distance_metric in self.supported_distance_metrics, msg

    def _validate_columns(self):
        cols = [self.date_col, self.units_col, self.outcome_metric]
        cols_not_found = list(set(cols) - set(self.data.columns))
        msg = f"Some of the column names provided were not found in data: {cols_not_found}"
        assert len(cols_not_found) == 0, msg

    def _coalesce_n_neighbors(self) -> int:
        """If n_neighbors is not specified, return self.default_n_neighbors if possible."""
        if self.n_neighbors is None:
            return min(self.n_control_units, self.default_n_neighbors)
        else:
            return self.n_neighbors

    def _validate_n_neighbors(self):
        """Assert self.n_neighbors <= self.n_control_units."""
        msg = (
            "n_neighborsr must be less than or equal to the number of control unit candidates.\n"
            f"n_neighbors: {self.n_neighbors}\n"
            f"n_control_units: {self.n_control_units}"
        )
        assert self.n_neighbors <= self.n_control_units, msg

    def _get_pre_intervention_data(self, data):
        """Subset data to the pre intervention period."""
        return data.query(f"{self.date_col} <= '{self.event_date}'")

    def dist_factory(self, data: pd.DataFrame) -> Callable[[str, str], float]:
        """Generate euclidean or dynamic time warping distance method."""
        if self.distance_metric == "dtw":

            def distance(cid1: str, cid2: str) -> Callable[[str, str], float]:
                """Distance function using DTWDistance."""
                return rust.dtw_distance(
                    data.loc[[cid1]][self.outcome_metric].to_numpy(),
                    data.loc[[cid2]][self.outcome_metric].to_numpy(),
                )

        elif self.distance_metric == "euclidean":

            def distance(cid1: str, cid2: str) -> Callable[[str, str], float]:
                """Distance function using euclidean distance."""
                return euclid_distance(
                    data.loc[cid1][self.outcome_metric],
                    data.loc[cid2][self.outcome_metric],
                )

        return distance

    def get_control_units(self):
        """Find control units nearest to the given test unit determined by the distance metric."""
        logging.info("Searching for best control units.")
        if self.n_neighbors == self.n_control_units:
            msg = (
                f"n_neighbors ({self.n_neighbors}) is equal to the number of available control "
                f"units ({self.n_control_units}). There is nothing for Kirby to do. Returning all "
                "available control units."
            )
            logging.info(colors.yellow(msg))

            # Output results as instance attributes
            self.unit_recs = self.control_units
            self.df_exp = self.data.reset_index()

        else:
            # Subset data to pre intervention data and generate the distance method
            pre_int_data = self._get_pre_intervention_data(self.data)
            distance_metric = self.dist_factory(pre_int_data)

            # Get the n closest control units for each test unit
            unit_recs = []
            logging.info("Iterating over test units and control units.")
            for test_unit in self.test_units:
                dist_list = []
                for control_unit in self.control_units:
                    dist_list.append(distance_metric(test_unit, control_unit))
                df_dist = pd.DataFrame(
                    {"controls": self.control_units, "dist": dist_list}
                )
                top_n = list(
                    df_dist.sort_values("dist").head(self.n_neighbors)["controls"]
                )
                unit_recs += top_n

            # Add in the test units
            unit_recs += self.test_units

            logging.info("Done searching for control units.")

            # Remove duplicates
            unit_recs = list(set(unit_recs))

            # Subset resulting data frame
            df_exp = self.data.reset_index()
            df_exp = df_exp[df_exp[self.units_col].isin(unit_recs)]

            # Output results as instance attributes
            self.unit_recs = unit_recs
            self.df_exp = df_exp
