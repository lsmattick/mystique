"""Base class for Mystique products."""
from datetime import datetime
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


class MystiqueBase():
    """Base class for Mystique products."""

    # Minimum number of days to have in the pre intervention data.
    min_pre_int_days = 5

    def _preprocess_data(self, data: pd.DataFrame, set_index=False) -> pd.DataFrame:
        """Preprocess data.

        Enforce the units column is of type str and the date column is of type pandas datetime.
        """
        data[self.units_col] = data.copy()[self.units_col].astype(str)
        if not is_datetime(data[self.date_col]):
            data[self.date_col] = pd.to_datetime(data.copy()[self.date_col])

        data = data.sort_values([self.units_col, self.date_col])
        if set_index:
            data = data.set_index(self.units_col)

        return data

    def _validate_test_units(self, index: bool = False):
        """Assert the given test units are found in the experiment data."""
        if not index:
            not_found = list(set(self.test_units) - set(self.data[self.units_col].unique()))

        else:
            not_found = list(set(self.test_units) - set(self.data.index))

        msg = f"Some of the test units provided could not be found: {not_found}"
        assert len(not_found) == 0, msg

    def _validate_event_date_format(self):
        """Assert the given event_date is of the correct format."""
        if self.event_date is not None:
            try:
                datetime.strptime(self.event_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Incorrect date format, please use YYYY-MM-DD")

    def _validate_event_date_in_range(self):
        """Assert the given event_date is within the range of dates found in the experiment data."""
        if self.event_date is not None:
            event_datetime = datetime.strptime(self.event_date, "%Y-%m-%d")
            min_date = self.data[self.date_col].min()
            max_date = self.data[self.date_col].max()
            msg = (
                "The given event date is not within range of the experiment data.\n"
                f"Event_date: {self.event_date} \n"
                f"Date range: {min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
            )
            assert event_datetime > min_date, msg

    def _validate_data_pre_event_date(self):
        """Assert there is sufficient data in the pre intervention period."""
        if self.event_date is not None:
            pre_intervention_data = self.data.query(f"{self.date_col} <= '{self.event_date}'")

            nrows = len(pre_intervention_data)
            msg = "No data before the event date."
            assert nrows > 0, msg

            ndays = pre_intervention_data[self.date_col].nunique()
            msg = "There must be at least 5 days of pre intervention data."
            assert ndays >= self.min_pre_int_days, msg

    def _coalesce_event_date(self):
        """Default to the max date found when event date is None."""
        if self.event_date is None:
            return datetime.strftime(self.data[self.date_col].max(), "%Y-%m-%d")
        else:
            return self.event_date
