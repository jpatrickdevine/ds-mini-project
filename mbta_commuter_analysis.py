"""
Title: MBTA Commuter Analysis
Author: [jpatrickdevine](https://github.com/jpatrickdevine)
Date created: 2026/06/26
Last modified: 2026/06/26
Description: This script analyzes MBTA commuter data, including estimated
  boardings, holidays, and weather conditions. It prepares the dataset
  for further analysis and modeling.
"""

# import numpy as np
import pandas as pd
# import keras
# from keras import layers
from matplotlib import pyplot as plt

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

# Sum the ridership by date
df_mbta = df_mbta.groupby('date').agg({'estimated_boardings': 'sum'}).reset_index()

# Write to csv file
df_mbta.to_csv("data/processed/mbta_commuter_rail.csv", index=False)

# -------------------------------------------------------------------------- #
# Clean up weather data
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

# Sort on date and line columns
df = df.sort_values(by='date').reset_index(drop=True)

# Create one day lag column for estimated_boardings and make integer type
df['estimated_boardings_lag_day'] = df['estimated_boardings'].shift(1).astype('Int64')

# Create 7 day lag column for estimated_boardings
df['estimated_boardings_lag_week'] = df['estimated_boardings'].shift(5).astype('Int64')

# Drop rows with NaN values in the lag columns
df = df.dropna(subset=['estimated_boardings_lag_day', 'estimated_boardings_lag_week'])

# Remove a row where is_holiday is True
df = df[~df['is_holiday']]

# Write to csv file
df.to_csv("data/final/dataset.csv", index=False)

print("Data wrangling complete. Processed dataset saved to data/final/dataset.csv")