from pathlib import Path
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm
from skyguard.config.paths import (
    RAW_DATA_DIR,
    HISTORICAL_PARQUET,
    HISTORICAL_CSV,
    PRESENT_PARQUET,
    PRESENT_CSV,
)

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
PRESENT_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

API_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = RAW_DATA_DIR


STATIONS = [
    {
        "station_id": "AWS_001",
        "station_name": "Delhi_Central",
        "latitude": 28.6139,
        "longitude": 77.2090,
    },
    {
        "station_id": "AWS_002",
        "station_name": "Gurugram",
        "latitude": 28.4595,
        "longitude": 77.0266,
    },
    {
        "station_id": "AWS_003",
        "station_name": "Noida",
        "latitude": 28.5355,
        "longitude": 77.3910,
    },
    {
        "station_id": "AWS_004",
        "station_name": "Faridabad",
        "latitude": 28.4089,
        "longitude": 77.3178,
    },
    {
        "station_id": "AWS_005",
        "station_name": "Ghaziabad",
        "latitude": 28.6692,
        "longitude": 77.4538,
    },
    {
        "station_id": "AWS_006",
        "station_name": "Sonipat",
        "latitude": 28.9931,
        "longitude": 77.0151,
    },
    {
        "station_id": "AWS_007",
        "station_name": "Rohtak",
        "latitude": 28.8955,
        "longitude": 76.6066,
    },
    {
        "station_id": "AWS_008",
        "station_name": "Meerut",
        "latitude": 28.9845,
        "longitude": 77.7064,
    },
    {
        "station_id": "AWS_009",
        "station_name": "Jhajjar",
        "latitude": 28.6067,
        "longitude": 76.6565,
    },
    {
        "station_id": "AWS_010",
        "station_name": "Greater_Noida",
        "latitude": 28.4744,
        "longitude": 77.5040,
    },
    {
        "station_id": "AWS_011",
        "station_name": "Bahadurgarh",
        "latitude": 28.6924,
        "longitude": 76.9356,
    },
    {
        "station_id": "AWS_012",
        "station_name": "Hapur",
        "latitude": 28.7306,
        "longitude": 77.7759,
    },
]


def fetch_station(station, start_date, end_date):
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "surface_pressure",
        ]),
        "timezone": "GMT",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["pressure_msl"],
        "surface_pressure": hourly["surface_pressure"],
    })

    df["station_id"] = station["station_id"]
    df["station_name"] = station["station_name"]
    df["latitude"] = station["latitude"]
    df["longitude"] = station["longitude"]

    return df


def fetch_and_save_dataset(stations, start_date, end_date, parquet_path, csv_path, dataset_name):
    """Fetch weather data for a date range and save in both formats."""
    all_data = []

    print(f"\nFetching {dataset_name}")
    print(f"Period:   {start_date} -> {end_date}")
    print(f"Stations: {len(stations)}\n")

    for station in tqdm(stations, desc="Downloading stations"):
        try:
            df = fetch_station(station, start_date, end_date)
            all_data.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"\nFailed: {station['station_name']}")
            print(f"Error: {e}")

    if not all_data:
        print(f"Warning: No data downloaded for {dataset_name}")
        return None

    weather_df = pd.concat(all_data, ignore_index=True)
    weather_df = weather_df[
        [
            "timestamp",
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "temperature",
            "humidity",
            "pressure",
            "surface_pressure",
        ]
    ]
    weather_df = weather_df.sort_values(
        ["timestamp", "station_id"]
    ).reset_index(drop=True)

    weather_df.to_parquet(parquet_path, index=False)
    weather_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 50)
    print(f"{dataset_name.upper()} CREATED SUCCESSFULLY")
    print("=" * 50)
    print(f"Shape:      {weather_df.shape}")
    print(f"Stations:   {weather_df['station_id'].nunique()}")
    print(f"Start:      {weather_df['timestamp'].min()}")
    print(f"End:        {weather_df['timestamp'].max()}")
    print(f"Saved to:   {parquet_path}")
    print(f"Saved to:   {csv_path}")
    print("\nMissing values:")
    print(weather_df.isnull().sum())
    print("\nRows per station:")
    print(weather_df.groupby("station_id").size())

    return weather_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    historical_df = fetch_and_save_dataset(
        STATIONS,
        START_DATE,
        END_DATE,
        HISTORICAL_PARQUET,
        HISTORICAL_CSV,
        "Historical Dataset (2023-2025)"
    )

    present_df = fetch_and_save_dataset(
        STATIONS,
        "2026-01-01",
        PRESENT_DATE,
        PRESENT_PARQUET,
        PRESENT_CSV,
        "Present Dataset (2026-Present)"
    )

    print("\n" + "=" * 70)
    print("DATA FETCH COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()