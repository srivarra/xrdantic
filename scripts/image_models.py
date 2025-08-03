"""
Image Processing Models using xrdantic.

This module demonstrates xrdantic models for multichannel images,
image datasets, and multi-resolution hierarchical structures.
"""

import numpy as np
import numpydantic.dtype as nptype

from xrdantic import Attr, Coordinate, Data, DataArray, Dataset, DataTree, Dim, Name

# Define image dimensions
Channel = Dim("channel")
Y = Dim("y")
X = Dim("x")
Image = Dim("image")

rng = np.random.default_rng(42)

# =============================================================================
# COORDINATE MODELS
# =============================================================================


class ChannelCoord(Coordinate):
    """Channel coordinate for multichannel images (e.g., RGB, fluorescence)."""

    data: Data[Channel, str]
    name: Name = "channel"
    long_name: Attr[str] = "image channels"


class YCoord(Coordinate):
    """Y-axis coordinate for image height."""

    data: Data[Y, int]
    name: Name = "y"
    units: Attr[str] = "pixels"
    long_name: Attr[str] = "image height"


class XCoord(Coordinate):
    """X-axis coordinate for image width."""

    data: Data[X, int]
    name: Name = "x"
    units: Attr[str] = "pixels"
    long_name: Attr[str] = "image width"


class ImageCoord(Coordinate):
    """Image ID coordinate for datasets with multiple images."""

    data: Data[Image, str]
    name: Name = "image"
    long_name: Attr[str] = "image identifier"


# =============================================================================
# DATAARRAY MODELS
# =============================================================================


class MultichannelImage(DataArray):
    """Multichannel image with Channel, Y, X dimensions."""

    data: Data[(Channel, Y, X), nptype.UInt8]
    channel: ChannelCoord
    y: YCoord
    x: XCoord
    name: Name = "image"
    units: Attr[str] = "intensity"
    long_name: Attr[str] = "multichannel image data"
    bit_depth: Attr[int] = 8


# =============================================================================
# DATASET MODELS
# =============================================================================


class ImageDataset(Dataset):
    """Dataset containing multichannel images."""

    images: MultichannelImage  # Can contain single or multiple images

    # Shared coordinates
    channel: ChannelCoord
    y: YCoord
    x: XCoord

    # Dataset attributes
    title: Attr[str] = "Multichannel Image Collection"
    acquisition_date: Attr[str] = "2024-01-15"
    microscope: Attr[str] = "Confocal LSM"
    magnification: Attr[str] = "63x"


# =============================================================================
# DATATREE MODELS - Multi-Resolution Pyramid
# =============================================================================


class MultiResolutionImages(DataTree):
    """Multi-resolution image pyramid with full and half resolution."""

    full_resolution: ImageDataset
    half_resolution: ImageDataset  # 1/2 size in X,Y, same C

    # Pyramid metadata
    pyramid_levels: Attr[int] = 2
    downsampling_method: Attr[str] = "bilinear"


class ImageAnalysisPyramid(DataTree):
    """Complete image analysis with raw data and processed versions."""

    raw: MultiResolutionImages
    processed: MultiResolutionImages

    # Analysis metadata
    processing_pipeline: Attr[str] = "denoising + enhancement"
    software_version: Attr[str] = "put whatever here"
    analysis_date: Attr[str] = "2024-01-15T10:30:00Z"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_sample_image(height: int = 256, width: int = 256, channels: list[str] | None = None) -> MultichannelImage:
    """Create a sample multichannel image."""
    if channels is None:
        channels = ["red", "green", "blue"]

    nchannels = len(channels)

    # Create coordinates
    channel_coord = ChannelCoord(data=np.array(channels), name="channel")
    y_coord = YCoord(data=np.arange(height), name="y")
    x_coord = XCoord(data=np.arange(width), name="x")

    # Create sample image data (random noise pattern)
    image_data = rng.integers(0, 256, size=(nchannels, height, width), dtype=np.uint8)

    return MultichannelImage(data=image_data, channel=channel_coord, y=y_coord, x=x_coord)


def create_downsampled_image(original: MultichannelImage, factor: int = 2) -> MultichannelImage:
    """Create a downsampled version of an image."""
    # Get original data and coordinates
    orig_data = original.data
    orig_channels = original.channel.data

    # Downsample by taking every nth pixel
    downsampled_data = orig_data[:, ::factor, ::factor]
    new_height, new_width = downsampled_data.shape[1], downsampled_data.shape[2]

    # Create new coordinates
    channel_coord = ChannelCoord(data=orig_channels, name="channel")
    y_coord = YCoord(data=np.arange(new_height), name="y")
    x_coord = XCoord(data=np.arange(new_width), name="x")

    return MultichannelImage(data=downsampled_data, channel=channel_coord, y=y_coord, x=x_coord)


def create_image_pyramid(height: int = 256, width: int = 256) -> MultiResolutionImages:
    """Create a multi-resolution image pyramid."""
    # Create full resolution image
    full_res_image = create_sample_image(height, width, ["red", "green", "blue"])

    # Wrap in dataset
    full_res_dataset = ImageDataset(
        images=full_res_image, channel=full_res_image.channel, y=full_res_image.y, x=full_res_image.x
    )

    # Create half resolution version
    half_res_image = create_downsampled_image(full_res_image, factor=2)

    # Wrap in dataset
    half_res_dataset = ImageDataset(
        images=half_res_image, channel=half_res_image.channel, y=half_res_image.y, x=half_res_image.x
    )

    return MultiResolutionImages(full_resolution=full_res_dataset, half_resolution=half_res_dataset)


def create_analysis_pyramid() -> ImageAnalysisPyramid:
    """Create a complete image analysis pyramid with raw and processed versions."""
    # Create raw image pyramid
    raw_pyramid = create_image_pyramid(512, 512)

    # Create "processed" version (just different data for demo)
    processed_pyramid = create_image_pyramid(512, 512)  # This will have different random data

    return ImageAnalysisPyramid(raw=raw_pyramid, processed=processed_pyramid)


if __name__ == "__main__":
    print("🖼️  Creating image processing models...")

    # 1. DataArray: Single multichannel image
    print("\n1. DataArray Example:")
    image = create_sample_image(128, 128, ["red", "green", "blue"])
    print(f"   Image shape: {image.data.shape} (channels={len(image.channel.data)})")
    print(f"   Channels: {list(image.channel.data)}")

    # 2. Dataset: Multiple images
    print("\n2. Dataset Example:")
    dataset_structure = ImageDataset.get_model_dataarray_fields()
    print(f"   Dataset contains: {list(dataset_structure.keys())}")
    print("   (Note: Dataset naturally handles multiple images via the images field)")

    # 3. DataTree: Multi-resolution pyramid
    print("\n3. DataTree Example:")
    pyramid = create_image_pyramid(256, 256)
    print(f"   Pyramid levels: {list(pyramid.get_dataset_model_fields().keys())}")
    print(f"   Full resolution: {pyramid.full_resolution.images.data.shape}")
    print(f"   Half resolution: {pyramid.half_resolution.images.data.shape}")

    # 4. Complex DataTree: Analysis pipeline
    print("\n4. Complex DataTree Example:")
    analysis = create_analysis_pyramid()
    print(f"   Analysis pipeline: {list(analysis.get_datatree_model_fields().keys())}")

    # Convert to xarray for inspection
    print("\n5. Conversion to xarray:")
    xr_image = image.to_xarray()
    print(f"   xarray DataArray dims: {xr_image.dims}")
    print(f"   xarray DataArray shape: {xr_image.shape}")
    print(f"   xarray coords: {list(xr_image.coords)}")
