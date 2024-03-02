from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureTransformer:
    def __init__(self, config: dict):
        self.config = config

    def drop_id(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop te ID column"""
        data = data.drop("id", axis=1)
        return data

    def convert_age_days_to_years(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert the 'age' column from days to years"""
        data["age"] = data["age"] / 365
        return data

    def add_bmi_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add a body mass index column based on the height and weight columns"""
        data["bmi"] = data["weight"] / (data["height"] / 100) ** 2
        return data

    def drop_height_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop the weight and height columns"""
        data = data.drop(["height", "weight"], axis=1)
        return data

    def cut_ap_samples(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Trim the pressure variables based on the config"""
        droped_indexes_max = data[
            (
                (data["ap_hi"] > self.config.ap_high_max)
                | (data["ap_lo"] > self.config.ap_low_max)
            )
        ].index
        droped_indexes_min = data[
            (
                (data["ap_hi"] < self.config.ap_high_min)
                | (data["ap_lo"] < self.config.ap_low_min)
            )
        ].index

        droped_indexes = droped_indexes_max.union(droped_indexes_min)
        data = data.drop(index=droped_indexes)
        return data, droped_indexes

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric columns"""
        data = StandardScaler().fit_transform(data)
        return data

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations in order"""
        data = data.copy()  # Create copy so that we can perform
                            # inplace operations
        # Transformation on whole data set
        data = self.drop_id(data)
        data = self.convert_age_days_to_years(data)
        data = self.add_bmi_column(data)
        data = self.drop_height_weight(data)

        # Trasformation that cuts samples
        data, droped_indexes = self.cut_ap_samples(data)

        # Scaler
        data = self.scale(data)

        return data, droped_indexes
