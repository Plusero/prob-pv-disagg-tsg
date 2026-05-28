import requests
import sys


def fetch_solcast_data(start_date='2010-06-30T00:00:00Z', end_date='2013-07-01T00:00:00Z', output_file='solcast_data.csv', location=1):
    if location == 1: # cluster 5 center
        latitude = -33.9
        longitude = 151.12
    elif location == 2:
        latitude = -33.86822
        longitude = 151.19678
    elif location == 3:
        latitude = -34.0168
        longitude = 151.11576
    else:
        raise ValueError("Invalid location")

    url = "https://api.solcast.com.au/data/historic/radiation_and_weather"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start': start_date,
        'end': end_date,
        'output_parameters': 'ghi,dhi,dni,clearsky_ghi,clearsky_dhi,clearsky_dni,air_temp,precipitation_rate,wind_speed_10m,wind_direction_10m,azimuth,zenith,cloud_opacity,weather_type',
        'format': 'csv',
        'api_key': 'YOUR_SOLCAST_API_KEY',
        "period": "PT30M"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    with open(output_file, 'w') as f:
        f.write(response.text)

    print(f"Data saved to: {output_file}")


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else '3'
    start = sys.argv[2] if len(sys.argv) > 2 else "2013-05-01T00:00:00Z"
    end = sys.argv[3] if len(sys.argv) > 3 else "2013-06-01T00:00:00Z"
    output = sys.argv[4] if len(
        sys.argv) > 4 else f"../data/ausgrid/ausgrid_meteo_location{location}_2011_Jan.csv"

    fetch_solcast_data(start_date=start, end_date=end,
                       output_file=output, location=int(location))
