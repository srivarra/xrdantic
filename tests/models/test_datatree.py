"""Tests for the DataTree model with property-based testing."""

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from xrdantic import Attr, Coordinate, Data, DataArray, Dataset, DataTree, Name
from xrdantic.types import Dim

from ..utils import DataGenerator  # noqa: TID252

# Define dimensions
X = Dim("x")
Y = Dim("y")
Time = Dim("time")
Layer = Dim("layer")

# ===== Hypothesis Strategies =====


@st.composite
def valid_3d_array_data(draw, min_size=2, max_size=6):
    """Generate valid 3D array data for DataTree testing."""
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    elements = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    shape = draw(array_shapes(min_dims=3, max_dims=3, min_side=min_size, max_side=max_size))
    return draw(arrays(dtype=dtype, shape=shape, elements=elements))


@st.composite
def coordinated_datatree_data(draw, min_size=2, max_size=5):
    """Generate coordinated data for DataTree testing."""
    time_size = draw(st.integers(min_value=min_size, max_value=max_size))
    y_size = draw(st.integers(min_value=min_size, max_value=max_size))
    x_size = draw(st.integers(min_value=min_size, max_value=max_size))
    layer_size = draw(st.integers(min_value=2, max_value=4))

    # Generate time-based data (for weather)
    temp_data = draw(
        arrays(
            dtype=np.float64,
            shape=(time_size, y_size, x_size),
            elements=st.floats(min_value=-30, max_value=40, allow_nan=False, allow_infinity=False),
        )
    )
    pressure_data = draw(
        arrays(
            dtype=np.float64,
            shape=(time_size, y_size, x_size),
            elements=st.floats(min_value=950, max_value=1050, allow_nan=False, allow_infinity=False),
        )
    )

    # Generate layer-based data (for radar)
    reflectivity_data = draw(
        arrays(
            dtype=np.float64,
            shape=(layer_size, y_size, x_size),
            elements=st.floats(min_value=-10, max_value=50, allow_nan=False, allow_infinity=False),
        )
    )

    return {
        "temp_data": temp_data,
        "pressure_data": pressure_data,
        "reflectivity_data": reflectivity_data,
        "time_size": time_size,
        "y_size": y_size,
        "x_size": x_size,
        "layer_size": layer_size,
    }


valid_observer_name = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(
        whitelist_categories=["Lu", "Ll", "Nd", "Pc"], min_codepoint=ord("A"), max_codepoint=ord("z")
    ),
)


# Define coordinate classes
class XCoord(Coordinate):
    data: Data[X, float]
    name: Name
    units: Attr[str] = "meters"


class YCoord(Coordinate):
    data: Data[Y, float]
    name: Name
    units: Attr[str] = "meters"


class TimeCoord(Coordinate):
    data: Data[Time, int]
    name: Name
    units: Attr[str] = "days"


class LayerCoord(Coordinate):
    data: Data[Layer, str]
    name: Name
    description: Attr[str] = "Layer name"


# Define DataArray classes
class Temperature(DataArray):
    data: Data[(Time, Y, X), float]
    x: XCoord
    y: YCoord
    time: TimeCoord
    name: Name
    units: Attr[str] = "celsius"


class Pressure(DataArray):
    data: Data[(Time, Y, X), float]
    x: XCoord
    y: YCoord
    time: TimeCoord
    name: Name
    units: Attr[str] = "hPa"


class Reflectivity(DataArray):
    data: Data[(Layer, Y, X), float]
    x: XCoord
    y: YCoord
    layer: LayerCoord
    name: Name
    units: Attr[str] = "dBZ"


# Define Dataset classes
class WeatherDataset(Dataset):
    temperature: Temperature
    pressure: Pressure
    x: XCoord
    y: YCoord
    time: TimeCoord
    station_name: Attr[str] = "Weather Station Alpha"


class RadarDataset(Dataset):
    reflectivity: Reflectivity
    x: XCoord
    y: YCoord
    layer: LayerCoord
    radar_id: Attr[str] = "Radar Unit 01"


# Define DataTree classes
class ObservationTree(DataTree):
    weather: WeatherDataset
    radar: RadarDataset
    x_global: XCoord  # Example of a root-level coordinate
    observer: Attr[str] = "Central Command"


class NestedTree(DataTree):
    obs1: ObservationTree
    obs2: ObservationTree
    run_id: Attr[str] = "Simulation_XYZ"


class TestDataTree:
    """Test the DataTree class."""

    @pytest.fixture
    def sample_coords(self):
        nx, ny, nt, nl = 3, 4, 5, 2
        x_coord = XCoord(data=np.linspace(0, 100, nx), name="longitude")
        y_coord = YCoord(data=np.linspace(-50, 50, ny), name="latitude")
        time_coord = TimeCoord(data=np.arange(nt), name="time_offset")
        layer_coord = LayerCoord(data=np.array([f"layer_{i}" for i in range(nl)]), name="band")
        return {
            "x": x_coord,
            "y": y_coord,
            "time": time_coord,
            "layer": layer_coord,
            "nx": nx,
            "ny": ny,
            "nt": nt,
            "nl": nl,
        }

    @pytest.fixture
    def sample_weather_dataset(self, sample_coords):
        coords = sample_coords
        nx, ny, nt = coords["nx"], coords["ny"], coords["nt"]

        temp_array = Temperature(
            data=DataGenerator.random_array((nt, ny, nx)),
            x=coords["x"],
            y=coords["y"],
            time=coords["time"],
            name="air_temp",
        )
        pressure_array = Pressure(
            data=DataGenerator.random_array((nt, ny, nx)) + 1000,
            x=coords["x"],
            y=coords["y"],
            time=coords["time"],
            name="surface_pressure",
        )
        return WeatherDataset(
            temperature=temp_array,
            pressure=pressure_array,
            x=coords["x"],
            y=coords["y"],
            time=coords["time"],
        )

    @pytest.fixture
    def sample_radar_dataset(self, sample_coords):
        coords = sample_coords
        nx, ny, nl = coords["nx"], coords["ny"], coords["nl"]

        refl_array = Reflectivity(
            data=DataGenerator.random_array((nl, ny, nx)) * 10,
            x=coords["x"],
            y=coords["y"],
            layer=coords["layer"],
            name="radar_reflectivity",
        )
        return RadarDataset(reflectivity=refl_array, x=coords["x"], y=coords["y"], layer=coords["layer"])

    def test_datatree_creation(self, sample_weather_dataset, sample_radar_dataset, sample_coords):
        """Test creating a DataTree instance."""
        obs_tree = ObservationTree(
            weather=sample_weather_dataset,
            radar=sample_radar_dataset,
            x_global=sample_coords["x"],
        )

        assert obs_tree.observer == "Central Command"
        assert obs_tree.weather.station_name == "Weather Station Alpha"
        assert obs_tree.radar.radar_id == "Radar Unit 01"
        assert obs_tree.x_global == sample_coords["x"]

        # Test nested DataTree creation
        nested = NestedTree(obs1=obs_tree, obs2=obs_tree)
        assert nested.run_id == "Simulation_XYZ"
        assert nested.obs1.weather.station_name == "Weather Station Alpha"

    @given(tree_data=coordinated_datatree_data(), observer=valid_observer_name)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_datatree_creation_property_based(self, tree_data, observer):
        """Property-based test for DataTree creation with various data shapes."""
        time_size = tree_data["time_size"]
        y_size = tree_data["y_size"]
        x_size = tree_data["x_size"]
        layer_size = tree_data["layer_size"]

        # Create coordinates
        x_coord = XCoord(data=np.linspace(0, 100, x_size), name="longitude")
        y_coord = YCoord(data=np.linspace(-50, 50, y_size), name="latitude")
        time_coord = TimeCoord(data=np.arange(time_size), name="time_offset")
        layer_coord = LayerCoord(data=np.array([f"layer_{i}" for i in range(layer_size)]), name="band")

        # Create data arrays
        temp_array = Temperature(
            data=tree_data["temp_data"],
            x=x_coord,
            y=y_coord,
            time=time_coord,
            name="air_temp",
        )
        pressure_array = Pressure(
            data=tree_data["pressure_data"],
            x=x_coord,
            y=y_coord,
            time=time_coord,
            name="surface_pressure",
        )
        refl_array = Reflectivity(
            data=tree_data["reflectivity_data"],
            x=x_coord,
            y=y_coord,
            layer=layer_coord,
            name="radar_reflectivity",
        )

        # Create datasets
        weather_dataset = WeatherDataset(
            temperature=temp_array,
            pressure=pressure_array,
            x=x_coord,
            y=y_coord,
            time=time_coord,
        )
        radar_dataset = RadarDataset(
            reflectivity=refl_array,
            x=x_coord,
            y=y_coord,
            layer=layer_coord,
        )

        # Create DataTree
        obs_tree = ObservationTree(
            weather=weather_dataset,
            radar=radar_dataset,
            x_global=x_coord,
            observer=observer,
        )

        # Verify properties
        assert obs_tree.observer == observer
        assert obs_tree.weather.station_name == "Weather Station Alpha"
        assert obs_tree.radar.radar_id == "Radar Unit 01"

        # Verify data shapes
        assert obs_tree.weather.temperature.data.shape == (time_size, y_size, x_size)
        assert obs_tree.weather.pressure.data.shape == (time_size, y_size, x_size)
        assert obs_tree.radar.reflectivity.data.shape == (layer_size, y_size, x_size)

        # Verify data integrity
        np.testing.assert_array_equal(obs_tree.weather.temperature.data, tree_data["temp_data"])
        np.testing.assert_array_equal(obs_tree.weather.pressure.data, tree_data["pressure_data"])
        np.testing.assert_array_equal(obs_tree.radar.reflectivity.data, tree_data["reflectivity_data"])

    def test_datatree_to_xarray(self, sample_weather_dataset, sample_radar_dataset, sample_coords):
        """Test converting DataTree to xarray.DataTree."""
        obs_tree_model = ObservationTree(
            weather=sample_weather_dataset,
            radar=sample_radar_dataset,
            x_global=sample_coords["x"],
            observer="Test Observer",
        )

        xr_tree = obs_tree_model.to_xarray()

        assert isinstance(xr_tree, xr.DataTree)
        assert xr_tree.name is None  # Default name for root
        assert "weather" in xr_tree
        assert "radar" in xr_tree
        assert isinstance(xr_tree["weather"].ds, xr.Dataset)  # Access underlying dataset via .ds
        assert isinstance(xr_tree["radar"].ds, xr.Dataset)

        # Check root attributes
        assert xr_tree.attrs["observer"] == "Test Observer"

        # Check root coordinates
        assert "x_global" in xr_tree.coords or "x" in xr_tree.coords
        if "x_global" in xr_tree.coords:
            np.testing.assert_array_equal(xr_tree.coords["x_global"].values, sample_coords["x"].data)
        else:
            np.testing.assert_array_equal(xr_tree.coords["x"].values, sample_coords["x"].data)

        # Check datasets
        assert xr_tree["weather"].ds.attrs["station_name"] == "Weather Station Alpha"
        assert "temperature" in xr_tree["weather"].ds
        assert "reflectivity" in xr_tree["radar"].ds

        # Test nested conversion
        nested_model = NestedTree(obs1=obs_tree_model, obs2=obs_tree_model, run_id="SimA")
        xr_nested_tree = nested_model.to_xarray()

        assert isinstance(xr_nested_tree, xr.DataTree)
        assert "obs1" in xr_nested_tree
        assert "obs2" in xr_nested_tree
        assert isinstance(xr_nested_tree["obs1"], xr.DataTree)
        assert xr_nested_tree.attrs["run_id"] == "SimA"
        assert "weather" in xr_nested_tree["obs1"]
        assert xr_nested_tree["obs1"]["weather"].ds.attrs["station_name"] == "Weather Station Alpha"

    @given(tree_data=coordinated_datatree_data(), observer=valid_observer_name)
    @settings(max_examples=8, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture])
    def test_datatree_to_xarray_conversion_property_based(self, tree_data, observer):
        """Property-based test for DataTree to xarray conversion with data integrity verification."""
        time_size = tree_data["time_size"]
        y_size = tree_data["y_size"]
        x_size = tree_data["x_size"]
        layer_size = tree_data["layer_size"]

        # Create coordinates
        x_coord = XCoord(data=np.linspace(0, 100, x_size), name="longitude")
        y_coord = YCoord(data=np.linspace(-50, 50, y_size), name="latitude")
        time_coord = TimeCoord(data=np.arange(time_size), name="time_offset")
        layer_coord = LayerCoord(data=np.array([f"layer_{i}" for i in range(layer_size)]), name="band")

        # Create data arrays
        temp_array = Temperature(
            data=tree_data["temp_data"],
            x=x_coord,
            y=y_coord,
            time=time_coord,
            name="air_temp",
        )
        pressure_array = Pressure(
            data=tree_data["pressure_data"],
            x=x_coord,
            y=y_coord,
            time=time_coord,
            name="surface_pressure",
        )
        refl_array = Reflectivity(
            data=tree_data["reflectivity_data"],
            x=x_coord,
            y=y_coord,
            layer=layer_coord,
            name="radar_reflectivity",
        )

        # Create datasets
        weather_dataset = WeatherDataset(
            temperature=temp_array,
            pressure=pressure_array,
            x=x_coord,
            y=y_coord,
            time=time_coord,
        )
        radar_dataset = RadarDataset(
            reflectivity=refl_array,
            x=x_coord,
            y=y_coord,
            layer=layer_coord,
        )

        # Create DataTree
        obs_tree_model = ObservationTree(
            weather=weather_dataset,
            radar=radar_dataset,
            x_global=x_coord,
            observer=observer,
        )

        xr_tree = obs_tree_model.to_xarray()

        # Verify structure
        assert isinstance(xr_tree, xr.DataTree)
        assert "weather" in xr_tree
        assert "radar" in xr_tree

        # Verify attributes
        assert xr_tree.attrs["observer"] == observer

        # Verify data integrity in the nested datasets
        assert "temperature" in xr_tree["weather"].ds
        assert "pressure" in xr_tree["weather"].ds
        assert "reflectivity" in xr_tree["radar"].ds

        # Verify data arrays match
        np.testing.assert_array_equal(xr_tree["weather"].ds["temperature"].values, tree_data["temp_data"])
        np.testing.assert_array_equal(xr_tree["weather"].ds["pressure"].values, tree_data["pressure_data"])
        np.testing.assert_array_equal(xr_tree["radar"].ds["reflectivity"].values, tree_data["reflectivity_data"])

    def test_datatree_validation_at_least_one_dataset_or_tree(self, sample_coords):
        """Test DataTree validation for content."""

        class EmptyTree(DataTree):
            description: Attr[str] = "An empty tree"
            x_coord: XCoord  # Root coord

        with pytest.raises(ValueError, match="DataTree must have at least one Dataset or DataTree field"):
            EmptyTree(x_coord=sample_coords["x"])

        # Should pass if it has a Dataset field (even if we pass a valid WeatherDataset instance)
        class TreeWithDataset(DataTree):
            data: WeatherDataset  # type: ignore
            description: Attr[str] = "Tree with one dataset"

        # Create a minimal weather dataset for testing
        temp_array = Temperature(
            data=DataGenerator.random_array((2, 2, 2), dtype=np.float32),  # type: ignore
            x=XCoord(data=np.array([0.0, 1.0]), name="x"),  # type: ignore
            y=YCoord(data=np.array([0.0, 1.0]), name="y"),  # type: ignore
            time=TimeCoord(data=np.array([0, 1]), name="time"),  # type: ignore
            name="temp",
        )
        pressure_array = Pressure(
            data=DataGenerator.random_array((2, 2, 2), dtype=np.float32),  # type: ignore
            x=XCoord(data=np.array([0.0, 1.0]), name="x"),  # type: ignore
            y=YCoord(data=np.array([0.0, 1.0]), name="y"),  # type: ignore
            time=TimeCoord(data=np.array([0, 1]), name="time"),  # type: ignore
            name="pressure",
        )
        test_weather_dataset = WeatherDataset(
            temperature=temp_array,
            pressure=pressure_array,
            x=XCoord(data=np.array([0.0, 1.0]), name="x"),  # type: ignore
            y=YCoord(data=np.array([0.0, 1.0]), name="y"),  # type: ignore
            time=TimeCoord(data=np.array([0, 1]), name="time"),  # type: ignore
        )

        # This should succeed because TreeWithDataset has a Dataset field
        tree_with_dataset = TreeWithDataset(data=test_weather_dataset)
        assert tree_with_dataset.description == "Tree with one dataset"

        # Should pass if it has another DataTree field
        class TreeWithTree(DataTree):
            child: ObservationTree  # type: ignore
            description: Attr[str] = "Tree with a child tree"

        # Create a minimal observation tree for testing
        test_obs_tree = ObservationTree(
            weather=test_weather_dataset,
            radar=RadarDataset(
                reflectivity=Reflectivity(
                    data=DataGenerator.random_array((2, 2, 2), dtype=np.float32),  # type: ignore
                    x=XCoord(data=np.array([0.0, 1.0]), name="x"),  # type: ignore
                    y=YCoord(data=np.array([0.0, 1.0]), name="y"),  # type: ignore
                    layer=LayerCoord(data=np.array(["a", "b"]), name="layer"),  # type: ignore
                    name="refl",
                ),
                x=XCoord(data=np.array([0.0, 1.0]), name="x"),  # type: ignore
                y=YCoord(data=np.array([0.0, 1.0]), name="y"),  # type: ignore
                layer=LayerCoord(data=np.array(["a", "b"]), name="layer"),  # type: ignore
            ),
            x_global=XCoord(data=np.array([0.0, 1.0]), name="x"),  # type: ignore
        )

        # This should succeed because TreeWithTree has a DataTree field
        tree_with_tree = TreeWithTree(child=test_obs_tree)
        assert tree_with_tree.description == "Tree with a child tree"

    @given(observer_name=valid_observer_name)
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_datatree_validation_error_property_based(self, observer_name, sample_coords):
        """Property-based test for DataTree validation errors."""

        class EmptyTree(DataTree):
            description: Attr[str] = "An empty tree"  # type: ignore
            x_coord: XCoord  # Root coord
            observer: Attr[str] = "test"  # type: ignore

        with pytest.raises(ValueError, match="DataTree must have at least one Dataset or DataTree field"):
            EmptyTree(x_coord=sample_coords["x"], observer=observer_name)

    def test_datatree_field_introspection(self, sample_weather_dataset, sample_radar_dataset, sample_coords):
        """Test field introspection methods for DataTree."""
        obs_tree = ObservationTree(
            weather=sample_weather_dataset,
            radar=sample_radar_dataset,
            x_global=sample_coords["x"],
        )

        dataset_fields = obs_tree.get_dataset_model_fields()
        datatree_fields = obs_tree.get_datatree_model_fields()
        coord_fields = obs_tree.get_coordinate_model_fields()  # Root-level coordinates
        attr_fields = obs_tree.get_attr_fields()

        assert len(dataset_fields) == 2
        assert "weather" in dataset_fields
        assert "radar" in dataset_fields

        assert len(datatree_fields) == 0  # No nested DataTree models directly in ObservationTree

        assert len(coord_fields) == 1
        assert "x_global" in coord_fields

        assert len(attr_fields) == 1
        assert "observer" in attr_fields

        # Test class methods
        model_dataset_fields = ObservationTree.get_model_dataset_fields()
        assert "weather" in model_dataset_fields
        assert model_dataset_fields["weather"].annotation == WeatherDataset

        model_datatree_fields = ObservationTree.get_model_datatree_fields()
        assert len(model_datatree_fields) == 0

        # Test with NestedTree
        nested_tree_model = NestedTree(obs1=obs_tree, obs2=obs_tree)
        nested_dataset_fields = nested_tree_model.get_dataset_model_fields()
        nested_datatree_fields = nested_tree_model.get_datatree_model_fields()

        assert len(nested_dataset_fields) == 0  # No direct Dataset fields
        assert len(nested_datatree_fields) == 2
        assert "obs1" in nested_datatree_fields
        assert "obs2" in nested_datatree_fields
        assert nested_datatree_fields["obs1"].annotation == ObservationTree

    @given(tree_data=coordinated_datatree_data(), observer=valid_observer_name)
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_datatree_field_introspection_property_based(self, tree_data, observer):
        """Property-based test for DataTree field introspection methods."""
        time_size = tree_data["time_size"]
        y_size = tree_data["y_size"]
        x_size = tree_data["x_size"]
        layer_size = tree_data["layer_size"]

        # Create coordinates
        x_coord = XCoord(data=np.linspace(0, 100, x_size), name="longitude")
        y_coord = YCoord(data=np.linspace(-50, 50, y_size), name="latitude")
        time_coord = TimeCoord(data=np.arange(time_size), name="time_offset")
        layer_coord = LayerCoord(data=np.array([f"layer_{i}" for i in range(layer_size)]), name="band")

        # Create minimal datasets for testing
        temp_array = Temperature(
            data=tree_data["temp_data"],
            x=x_coord,
            y=y_coord,
            time=time_coord,
            name="air_temp",
        )
        pressure_array = Pressure(
            data=tree_data["pressure_data"],
            x=x_coord,
            y=y_coord,
            time=time_coord,
            name="surface_pressure",
        )
        weather_dataset = WeatherDataset(
            temperature=temp_array,
            pressure=pressure_array,
            x=x_coord,
            y=y_coord,
            time=time_coord,
        )

        refl_array = Reflectivity(
            data=tree_data["reflectivity_data"],
            x=x_coord,
            y=y_coord,
            layer=layer_coord,
            name="radar_reflectivity",
        )
        radar_dataset = RadarDataset(
            reflectivity=refl_array,
            x=x_coord,
            y=y_coord,
            layer=layer_coord,
        )

        obs_tree = ObservationTree(
            weather=weather_dataset,
            radar=radar_dataset,
            x_global=x_coord,
            observer=observer,
        )

        # Test field categorization consistency
        dataset_fields = obs_tree.get_dataset_model_fields()
        coord_fields = obs_tree.get_coordinate_model_fields()
        attr_fields = obs_tree.get_attr_fields()

        assert len(dataset_fields) == 2
        assert "weather" in dataset_fields and "radar" in dataset_fields

        assert len(coord_fields) == 1
        assert "x_global" in coord_fields

        assert len(attr_fields) == 1
        assert "observer" in attr_fields

    def test_datatree_new_factory_method(self, sample_weather_dataset, sample_radar_dataset, sample_coords):
        """Test the .new() factory method for DataTree."""
        xr_tree = ObservationTree.new(
            weather=sample_weather_dataset,
            radar=sample_radar_dataset,
            x_global=sample_coords["x"],
            observer="FactoryMade",
        )

        assert isinstance(xr_tree, xr.DataTree)
        assert xr_tree.attrs["observer"] == "FactoryMade"
        assert "weather" in xr_tree
        assert "radar" in xr_tree
        assert "temperature" in xr_tree["weather"].ds

    def test_datatree_with_root_name(self, sample_weather_dataset, sample_radar_dataset, sample_coords):
        """Test DataTree creation with a root name."""

        class NamedRootTree(DataTree):
            weather: WeatherDataset  # type: ignore
            name: Name  # Add Name field for the root
            description: Attr[str] = "Tree with a named root"

        tree_model = NamedRootTree(weather=sample_weather_dataset, name="MyExperiment")
        xr_tree = tree_model.to_xarray()

        assert isinstance(xr_tree, xr.DataTree)
        assert xr_tree.name == "MyExperiment"
        assert "weather" in xr_tree
        assert xr_tree.attrs["description"] == "Tree with a named root"
