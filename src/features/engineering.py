"""
Feature engineering module for AI4I 2020 Predictive Maintenance.
Adds domain-informed physical operating features without mutating input DataFrames.
"""

import math
from typing import Optional
import pandas as pd

from src.features.schema import (
    FEATURE_TEMP_DIFF,
    FEATURE_MECH_POWER,
    RAW_NUMERIC_FEATURES,
)


def compute_temperature_difference(df: pd.DataFrame) -> pd.Series:
    """
    Computes the temperature differential between process and ambient air.

    Formula:
        temperature_difference = Process temperature [K] - Air temperature [K]

    Physical Unit:
        Kelvin [K]

    Description:
        Reflects the thermal gradient between the active machining process and ambient air.
        In the AI4I simulation, heat dissipation failures (HDF) are triggered when this
        thermal gradient drops below 8.6 K under low spindle speeds.

    Args:
        df: DataFrame containing 'Process temperature [K]' and 'Air temperature [K]'.

    Returns:
        pd.Series: Calculated temperature difference in Kelvin.

    Raises:
        KeyError: If required temperature columns are missing.
    """
    col_process = "Process temperature [K]"
    col_air = "Air temperature [K]"

    missing = [c for c in (col_process, col_air) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot compute temperature difference. Missing required columns: {missing}"
        )

    return df[col_process] - df[col_air]


def compute_mechanical_power(df: pd.DataFrame) -> pd.Series:
    """
    Computes shaft mechanical power transmitted by the spindle.

    Formula:
        mechanical_power_w = Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60)

    Physical Unit:
        Watts [W] (Joule/second = N*m/s)

    Description:
        Represents the shaft mechanical power delivered through the machine spindle shaft,
        calculated as the product of mechanical torque and rotational angular velocity
        (converted from rpm to radians per second via 2*pi/60).
        NOTE: This is shaft mechanical power, NOT electrical input power to the motor drive.
        In the AI4I simulation, power failures (PWF) are triggered when shaft mechanical power
        falls outside the acceptable window [3,500 W, 9,000 W].

    Args:
        df: DataFrame containing 'Torque [Nm]' and 'Rotational speed [rpm]'.

    Returns:
        pd.Series: Calculated shaft mechanical power in Watts.

    Raises:
        KeyError: If required torque or rotational speed columns are missing.
    """
    col_torque = "Torque [Nm]"
    col_speed = "Rotational speed [rpm]"

    missing = [c for c in (col_torque, col_speed) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot compute mechanical power. Missing required columns: {missing}"
        )

    angular_velocity_rad_s = df[col_speed] * (2.0 * math.pi / 60.0)
    return df[col_torque] * angular_velocity_rad_s


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain feature engineering to create physical diagnostic indicators.

    Guarantees:
        - The caller's DataFrame is NEVER mutated (operates on a deep copy).
        - No rows are silently dropped or altered.
        - Target labels and existing features remain intact.

    Added Features:
        1. temperature_difference (Kelvin [K])
        2. mechanical_power_w (Watts [W] - shaft mechanical power)

    Args:
        df: Input DataFrame containing operating sensor columns.

    Returns:
        pd.DataFrame: A new DataFrame containing original columns plus engineered features.

    Raises:
        KeyError: If columns required to derive the features are missing.
    """
    # Create an explicit copy to guarantee immutability of the input DataFrame
    engineered = df.copy()

    # Compute physical features
    engineered[FEATURE_TEMP_DIFF] = compute_temperature_difference(engineered)
    engineered[FEATURE_MECH_POWER] = compute_mechanical_power(engineered)

    return engineered

