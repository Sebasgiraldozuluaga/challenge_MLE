"""
Delay prediction model for flight delays.

This module implements a machine learning model to predict flight delays
based on flight information using XGBoost with class balancing.
"""
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb


class DelayModel:
    """
    A delay prediction model using XGBoost.

    This model predicts flight delays based on top 10 most important features
    identified through feature importance analysis. It uses class balancing
    (scale_pos_weight) to improve recall for the minority class (delayed flights).
    """

    # Top 10 most important features from feature importance analysis
    TOP_10_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    # Threshold in minutes to classify a flight as delayed
    DELAY_THRESHOLD_MINUTES = 15

    def __init__(self):
        """Initialize the DelayModel."""
        self._model = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        This method performs feature engineering including:
        - Generating the delay target variable
        - One-hot encoding categorical features
        - Selecting top 10 most important features

        Args:
            data (pd.DataFrame): Raw data with flight information.
            target_column (str, optional): If set, the target is returned.
                Should be "delay" for training.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
                If target_column is provided: (features, target) tuple
                If target_column is None: features only

        Raises:
            KeyError: If required columns are missing from the data.
        """
        # Create a copy to avoid modifying original data
        data = data.copy()

        # Generate delay target if needed and not present
        if target_column and 'delay' not in data.columns:
            data['delay'] = self._generate_delay(data)

        # Generate one-hot encoded features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        # Select only top 10 features, filling missing columns with zeros
        for feature in self.TOP_10_FEATURES:
            if feature not in features.columns:
                features[feature] = 0

        features = features[self.TOP_10_FEATURES]

        # Return features and target if target_column is specified
        if target_column:
            if target_column not in data.columns:
                raise KeyError(
                    f"Target column '{target_column}' not found in data"
                )
            target = data[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Uses XGBoost with scale_pos_weight to handle
        the imbalanced dataset (more non-delayed than delayed flights).

        Args:
            features (pd.DataFrame): Preprocessed features.
            target (pd.DataFrame): Target variable (delay).

        Raises:
            ValueError: If features or target are empty.
        """
        if features.empty or target.empty:
            raise ValueError("Features and target cannot be empty")

        # Calculate scale for balancing: ratio of negative to positive samples
        n_y0 = len(target[target.iloc[:, 0] == 0])
        n_y1 = len(target[target.iloc[:, 0] == 1])
        scale = n_y0 / n_y1

        # Initialize and train XGBoost model with class balancing
        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale
        )

        # Fit the model
        self._model.fit(features, target.iloc[:, 0])

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): Preprocessed features.

        Returns:
            List[int]: Predicted targets (0 = no delay, 1 = delay).

        Note:
            If model has not been fitted, returns default predictions (all 0s).
        """
        # If model hasn't been trained, return default predictions
        if self._model is None:
            return [0] * len(features)

        # Make predictions and convert to list of integers
        predictions = self._model.predict(features)
        return predictions.tolist()

    def _generate_delay(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate delay target variable from flight dates.

        A flight is considered delayed if the difference between
        actual departure (Fecha-O) and scheduled departure (Fecha-I)
        is greater than DELAY_THRESHOLD_MINUTES.

        Args:
            data (pd.DataFrame): Raw data with Fecha-I and Fecha-O columns.

        Returns:
            pd.Series: Binary delay indicator (1 = delayed, 0 = not delayed).

        Raises:
            KeyError: If required date columns are missing.
        """
        required_cols = ['Fecha-O', 'Fecha-I']
        for col in required_cols:
            if col not in data.columns:
                raise KeyError(f"Required column '{col}' not found in data")

        def get_min_diff(row):
            """Calculate minute difference between scheduled and actual."""
            try:
                fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
                fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
                return (fecha_o - fecha_i).total_seconds() / 60
            except (ValueError, TypeError):
                # If parsing fails, return 0 (no delay)
                return 0

        min_diff = data.apply(get_min_diff, axis=1)
        return np.where(min_diff > self.DELAY_THRESHOLD_MINUTES, 1, 0)
