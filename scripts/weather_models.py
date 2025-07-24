"""
Weather and Climate Models using xrdantic.

This module demonstrates how to create Pydantic-validated xarray data structures
for meteorological and climate data using xrdantic.
"""

import numpy as np
import numpydantic.dtype as npdtype

from xrdantic import Attr, Coordinate, Data, DataArray, Dataset, DataTree, Dim, Name

rng = np.random.default_rng(42)

# Define dimensions for weather data
Time = Dim("time")
Lat = Dim("lat")
Lon = Dim("lon")
Level = Dim("level")
Station = Dim("station")

# =============================================================================
# COORDINATE MODELS
# =============================================================================


class TimeCoord(Coordinate):
    """Time coordinate for weather data."""

    data: Data[Time, npdtype.Datetime64]
    name: Name
    units: Attr[str] = "days since 1900-01-01"
    calendar: Attr[str] = "gregorian"
    long_name: Attr[str] = "time"


class LatitudeCoord(Coordinate):
    """Latitude coordinate."""

    data: Data[Lat, npdtype.Float]
    name: Name = "latitude"
    units: Attr[str] = "degrees_north"
    long_name: Attr[str] = "latitude"
    valid_min: Attr[float] = -90.0
    valid_max: Attr[float] = 90.0


class LongitudeCoord(Coordinate):
    """Longitude coordinate."""

    data: Data[Lon, npdtype.Float]
    name: Name = "longitude"
    units: Attr[str] = "degrees_east"
    long_name: Attr[str] = "longitude"
    valid_min: Attr[float] = -180.0
    valid_max: Attr[float] = 180.0


class PressureLevelCoord(Coordinate):
    """Atmospheric pressure level coordinate."""

    data: Data[Level, npdtype.Float]
    name: Name = "level"
    units: Attr[str] = "hPa"
    long_name: Attr[str] = "pressure level"
    positive: Attr[str] = "down"


class StationCoord(Coordinate):
    """Weather station ID coordinate."""

    data: Data[Station, str]
    name: Name = "station"
    long_name: Attr[str] = "weather station identifier"


# =============================================================================
# DATAARRAY MODELS
# =============================================================================


class Temperature(DataArray):
    """2D surface temperature field."""

    data: Data[(Time, Lat, Lon), npdtype.Float]
    time: TimeCoord
    lat: LatitudeCoord
    lon: LongitudeCoord
    name: Name = "temperature"
    units: Attr[str] = "K"
    long_name: Attr[str] = "air temperature"
    standard_name: Attr[str] = "air_temperature"
    valid_min: Attr[float] = 200.0
    valid_max: Attr[float] = 330.0


class Precipitation(DataArray):
    """2D precipitation field."""

    data: Data[(Time, Lat, Lon), npdtype.Float]
    time: TimeCoord
    lat: LatitudeCoord
    lon: LongitudeCoord
    name: Name = "precipitation"
    units: Attr[str] = "mm/day"
    long_name: Attr[str] = "precipitation rate"
    standard_name: Attr[str] = "precipitation_flux"
    valid_min: Attr[float] = 0.0


class RelativeHumidity(DataArray):
    """3D relative humidity field."""

    data: Data[(Time, Level, Lat, Lon), npdtype.Float]
    time: TimeCoord
    level: PressureLevelCoord
    lat: LatitudeCoord
    lon: LongitudeCoord
    name: Name = "humidity"
    units: Attr[str] = "%"
    long_name: Attr[str] = "relative humidity"
    standard_name: Attr[str] = "relative_humidity"
    valid_min: Attr[float] = 0.0
    valid_max: Attr[float] = 100.0


class WindSpeed(DataArray):
    """Wind speed field."""

    data: Data[(Time, Level, Lat, Lon), npdtype.Float]
    time: TimeCoord
    level: PressureLevelCoord
    lat: LatitudeCoord
    lon: LongitudeCoord
    name: Name = "wind_speed"
    units: Attr[str] = "m/s"
    long_name: Attr[str] = "wind speed"
    standard_name: Attr[str] = "wind_speed"
    valid_min: Attr[float] = 0.0


class StationTemperature(DataArray):
    """Point temperature observations from weather stations."""

    data: Data[(Time, Station), npdtype.Float]
    time: TimeCoord
    station: StationCoord
    name: Name = "station_temp"
    units: Attr[str] = "celsius"
    long_name: Attr[str] = "station air temperature"
    measurement_height: Attr[str] = "2m"


# =============================================================================
# DATASET MODELS
# =============================================================================


class SurfaceWeatherDataset(Dataset):
    """Surface weather dataset with temperature and precipitation."""

    temperature: Temperature
    precipitation: Precipitation

    # Shared coordinates
    time: TimeCoord
    lat: LatitudeCoord
    lon: LongitudeCoord

    # Dataset attributes
    title: Attr[str] = "Surface Weather Analysis"
    institution: Attr[str] = "National Weather Service"
    source: Attr[str] = "reanalysis"
    conventions: Attr[str] = "CF-1.8"


class AtmosphericDataset(Dataset):
    """3D atmospheric dataset with humidity and wind."""

    humidity: RelativeHumidity
    wind_speed: WindSpeed

    # Shared coordinates
    time: TimeCoord
    level: PressureLevelCoord
    lat: LatitudeCoord
    lon: LongitudeCoord

    # Dataset attributes
    title: Attr[str] = "Atmospheric Reanalysis Data"
    institution: Attr[str] = "ECMWF"
    source: Attr[str] = "ERA5 reanalysis"
    vertical_coordinate: Attr[str] = "pressure"


class StationObservations(Dataset):
    """Weather station point observations."""

    temperature: StationTemperature

    # Shared coordinates
    time: TimeCoord
    station: StationCoord

    # Dataset attributes
    title: Attr[str] = "Surface Station Observations"
    institution: Attr[str] = "NOAA"
    source: Attr[str] = "automated weather stations"
    temporal_resolution: Attr[str] = "hourly"


# =============================================================================
# DATATREE MODELS
# =============================================================================


class WeatherAnalysisTree(DataTree):
    """Complete weather analysis with surface and atmospheric data."""

    surface: SurfaceWeatherDataset
    atmosphere: AtmosphericDataset

    # Root-level attributes
    analysis_time: Attr[str] = "2024-01-15T12:00:00Z"
    model_version: Attr[str] = "v2.1"
    grid_resolution: Attr[str] = "0.25 degrees"
    forecast_reference_time: Attr[str] = "2024-01-15T00:00:00Z"


class MultiSourceWeatherTree(DataTree):
    """Weather data from multiple sources and resolutions."""

    gridded_analysis: WeatherAnalysisTree
    station_obs: StationObservations

    # Root-level metadata
    project: Attr[str] = "Climate Monitoring Initiative"
    quality_control: Attr[str] = "automated + manual review"
    spatial_coverage: Attr[str] = "global"
    temporal_coverage: Attr[str] = "1979-present"


class ForecastTree(DataTree):
    """Nested forecast tree with different lead times."""

    short_range: SurfaceWeatherDataset  # 0-7 days
    medium_range: SurfaceWeatherDataset  # 8-15 days

    # Forecast metadata
    model_name: Attr[str] = "GFS"
    initialization_time: Attr[str] = "2024-01-15T00:00:00Z"
    forecast_type: Attr[str] = "deterministic"
    domain: Attr[str] = "global"


# =============================================================================
# FACTORY FUNCTIONS FOR EASY MODEL CREATION
# =============================================================================


def create_sample_temperature_data(nt: int = 10, nlat: int = 5, nlon: int = 8) -> Temperature:
    """Create sample temperature data for testing."""
    import pandas as pd

    # Create coordinates
    time_coord = TimeCoord(data=pd.date_range("2024-01-01", periods=nt, freq="D").values, name="time")
    lat_coord = LatitudeCoord(data=np.linspace(-45, 45, nlat), name="latitude")
    lon_coord = LongitudeCoord(data=np.linspace(-90, 90, nlon), name="longitude")

    # Create realistic temperature data (in Kelvin)
    temp_data = 273.15 + 20 + 10 * rng.standard_normal(nt, nlat, nlon)

    return Temperature(data=temp_data, time=time_coord, lat=lat_coord, lon=lon_coord)


def create_sample_weather_dataset(nt: int = 10, nlat: int = 5, nlon: int = 8) -> SurfaceWeatherDataset:
    """Create a complete sample weather dataset."""
    import pandas as pd

    # Create shared coordinates
    time_coord = TimeCoord(data=pd.date_range("2024-01-01", periods=nt, freq="D").values, name="time")
    lat_coord = LatitudeCoord(data=np.linspace(-45, 45, nlat), name="latitude")
    lon_coord = LongitudeCoord(data=np.linspace(-90, 90, nlon), name="longitude")

    # Create temperature data
    temp_data = 273.15 + 20 + 10 * rng.standard_normal(nt, nlat, nlon)
    temperature = Temperature(data=temp_data, time=time_coord, lat=lat_coord, lon=lon_coord)

    # Create precipitation data
    precip_data = np.maximum(0, rng.exponential(2.0, (nt, nlat, nlon)))
    precipitation = Precipitation(data=precip_data, time=time_coord, lat=lat_coord, lon=lon_coord)

    return SurfaceWeatherDataset(
        temperature=temperature, precipitation=precipitation, time=time_coord, lat=lat_coord, lon=lon_coord
    )


def create_sample_forecast_tree(nt_short: int = 7, nt_medium: int = 8) -> ForecastTree:
    """Create a sample forecast tree with short and medium range forecasts."""
    nlat, nlon = 5, 8

    # Create short-range forecast
    short_range = create_sample_weather_dataset(nt_short, nlat, nlon)

    # Create medium-range forecast (slightly different characteristics)
    medium_range = create_sample_weather_dataset(nt_medium, nlat, nlon)

    return ForecastTree(short_range=short_range, medium_range=medium_range)


if __name__ == "__main__":
    # Example usage
    print("Creating sample weather models...")

    # Create a temperature field
    temp = create_sample_temperature_data()
    print(f"Temperature DataArray shape: {temp.data.shape}")

    # Create a complete weather dataset
    weather_ds = create_sample_weather_dataset()
    print(f"Weather Dataset variables: {list(weather_ds.get_dataarray_model_fields().keys())}")

    # Create a forecast tree
    forecast = create_sample_forecast_tree()
    print(f"Forecast Tree structure: {list(forecast.get_dataset_model_fields().keys())}")

    # Convert to xarray for inspection
    xr_weather = weather_ds.to_xarray()
    print(f"Converted to xarray Dataset with variables: {list(xr_weather.data_vars)}")
