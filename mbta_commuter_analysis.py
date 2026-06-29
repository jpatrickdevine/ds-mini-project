"""
Title: MBTA Commuter Analysis
Author: [jpatrickdevine](https://github.com/jpatrickdevine)
Date created: 2026/06/26
Last modified: 2026/06/28
Description: This script analyzes MBTA commuter data, including estimated
  boardings, holidays, and weather conditions. It prepares the dataset
  for further analysis and modeling.
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

"""
## Data Wrangling
"""
# -------------------------------------------------------------------------- #
# Clean up MBTA commuter rail data
# -------------------------------------------------------------------------- #
# Read in commuter rail by line csv file
df_mbta = pd.read_csv("data/raw/Commuter_Rail_by_Line.csv")

# Convert the date column to datetime object
df_mbta['servicedate'] = pd.to_datetime(df_mbta['servicedate'])

# Rename servicedate column to date for consistency
df_mbta.rename(columns={'servicedate': 'date'}, inplace=True)

# Clean up data - keep only servicedates from 2024-2026
df_mbta = df_mbta[(df_mbta['date'] >= '2024-01-01')]

# Remove Saturday and Sunday servicedates
df_mbta = df_mbta[df_mbta['date'].dt.dayofweek < 5]

# Rename lines with special characters
df_mbta['line'] = df_mbta['line'].str.replace(' ', '_').str.replace('/', '__')

# Sum the ridership by date
df_mbta = df_mbta.groupby('date').agg({'estimated_boardings': 'sum'}).reset_index()

# Write to csv file
df_mbta.to_csv("data/processed/mbta_commuter_rail.csv", index=False)

# -------------------------------------------------------------------------- #
# Clean up and merge holiday data
# -------------------------------------------------------------------------- #
# Read in and join holiday csv files
df_fed = pd.read_csv("data/raw/fed-holidays.csv")
df_nyse = pd.read_csv("data/raw/nyse-holidays.csv")
df_pat = pd.read_csv("data/raw/patriots-day.csv")

# Convert the date columns to datetime objects
df_fed['date'] = pd.to_datetime(df_fed['date'])
df_nyse['date'] = pd.to_datetime(df_nyse['date'])
df_pat['date'] = pd.to_datetime(df_pat['date'])

# Join the dataframes on the date column
df_merged = pd.merge(df_fed, df_nyse, on='date', how='outer', suffixes=('_fed', '_nyse'))
df_merged = pd.merge(df_merged, df_pat, on='date', how='outer')

# Rename holiday column in patriots-day dataframe to avoid confusion
df_merged.rename(columns={'holiday': 'holiday_pat'}, inplace=True)

# Write to csv file
df_merged.to_csv("data/processed/all_holidays.csv", index=False)

# -------------------------------------------------------------------------- #
# Clean up Boston weather data
# -------------------------------------------------------------------------- #
# Read in Boston weather data csv file
df_weather = pd.read_csv("data/raw/4343601.csv")

# Filter out rows where STATION is not USW00014739 (Boston Logan International Airport)
df_weather = df_weather[df_weather['STATION'] == 'USW00014739']

# Only keep columns DATE, PRCP, SNOW, TMAX, TMIN
df_weather = df_weather[['DATE', 'PRCP', 'SNOW', 'TMAX', 'TMIN']]

# Make column names lowercase
df_weather.columns = df_weather.columns.str.lower()

# Write to csv file
df_weather.to_csv("data/processed/boston_weather.csv", index=False)

# -------------------------------------------------------------------------- #
# Merge the dataframes and create new features
# -------------------------------------------------------------------------- #
# Read in processed files
df_mbta = pd.read_csv("data/processed/mbta_commuter_rail.csv")
df_holidays = pd.read_csv("data/processed/all_holidays.csv")
df_weather = pd.read_csv("data/processed/boston_weather.csv")

# Convert the date column to datetime object
df_mbta['date'] = pd.to_datetime(df_mbta['date'])
df_holidays['date'] = pd.to_datetime(df_holidays['date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])

# Create new data frame from mbta data
df = df_mbta.copy()

# # Look up holidays table and on df create a column called is_holiday that is True if the date is a holiday and False if it is not
df['is_holiday'] = df['date'].isin(df_holidays['date'])

# Merge the weather data into the new data frame on the date column
df = pd.merge(df, df_weather, on='date', how='left')

# Create a new column called day_of_week that contains the day of the week for each date
df['day_of_week'] = df['date'].dt.day_name()

# Create a new column called month that contains the month for each date
df['month'] = df['date'].dt.month_name()

# Create a new column called year that contains the year for each date
df['year'] = df['date'].dt.year

# Sort on date
df = df.sort_values(by='date').reset_index(drop=True)

# Create one day lag column for estimated_boardings and make integer type
df['estimated_boardings_lag_day'] = df['estimated_boardings'].shift(1).astype('Int64')

# Create 5 day lag column for estimated_boardings
df['estimated_boardings_lag_week'] = df['estimated_boardings'].shift(5).astype('Int64')

# Drop rows with NaN values in the lag columns
df = df.dropna(subset=['estimated_boardings_lag_day', 'estimated_boardings_lag_week'])

# Remove a row where is_holiday is True
df = df[~df['is_holiday']]

# Write to csv file
df.to_csv("data/final/dataset.csv", index=False)

print("Data wrangling complete. Processed dataset saved to data/final/dataset.csv")

# -------------------------------------------------------------------------- #
# From LLM...as I'm still learning, and I ran out of time, I prompted Github
# Copilot for help. I asked to apply a Keras regression model to the final
# dataset. I then asked to predict the number of estimated boardings for a 
# specific date, and prompted a few times to refine the predictions.
# -------------------------------------------------------------------------- #

DATA_PATH = "data/final/dataset.csv"
FEATURE_COLUMNS = [
    "prcp",
    "snow",
    "tmax",
    "tmin",
    "year",
    "estimated_boardings_lag_day",
    "estimated_boardings_lag_week",
    "day_of_week",
    "month",
]


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    X = pd.get_dummies(X, columns=["day_of_week", "month"], dtype=float)
    return X.astype(float)


def predict_for_date(model: keras.Model,
                     scaler: StandardScaler,
                     target_scaler: StandardScaler,
                     df: pd.DataFrame,
                     target_date: str):
    target = pd.Timestamp(target_date)
    row = df[df["date"] == target]
    if row.empty:
        raise ValueError(f"No rows found for date {target_date}")

    row_features = prepare_features(row)
    row_features = row_features.reindex(columns=scaler.feature_names_in_, fill_value=0.0)
    scaled_row = scaler.transform(row_features)
    prediction_scaled = model.predict(scaled_row, verbose=0)[0][0]
    prediction = target_scaler.inverse_transform(np.array([[prediction_scaled]])).ravel()[0]

    actual = float(row["estimated_boardings"].iloc[0])
    print(f"Predicted estimated boardings for {target_date}: {prediction:,.0f}")
    print(f"Actual estimated boardings: {actual:,.0f}")


## This part would normally be at the top of a main function, but I kept it at the bottom for now.
parser = argparse.ArgumentParser(description="Train a Keras regression model and predict boardings for a date")
parser.add_argument("--date", type=str, default=None, help="Date to predict in YYYY-MM-DD format")
parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to the prepared CSV file")
args = parser.parse_args()

df = pd.read_csv(args.data_path)
df["date"] = pd.to_datetime(df["date"])

X = prepare_features(df)
y = df["estimated_boardings"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.to_frame()).ravel()
y_test_scaled = target_scaler.transform(y_test.to_frame()).ravel()

model = build_model(X_train.shape[1])

model.fit(
    X_train,
    y_train_scaled,
    validation_split=0.1,
    epochs=200,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
    verbose=0,
)

test_loss, test_mae = model.evaluate(X_test, y_test_scaled, verbose=0)
preds_scaled = model.predict(X_test, verbose=0).ravel()
preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

print(f"Test MSE: {test_loss:.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, preds):.2f}")

if args.date:
    predict_for_date(model, scaler, target_scaler, df, args.date)
