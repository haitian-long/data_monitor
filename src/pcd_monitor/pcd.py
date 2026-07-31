from concurrent.futures import ThreadPoolExecutor
import math
import struct
from pathlib import Path

import numpy as np


DISPLAY_Z_PERCENTILES = (1.0, 99.0)
BEV_RASTER_WORKERS = 4


class PcdError(RuntimeError):
    pass


def validate_pcd_file(input_path):
    """Validate and return one PCD file path."""
    path = Path(input_path).expanduser()
    if not path.exists():
        raise PcdError("PCD input file does not exist: {}".format(path))
    if not path.is_file():
        raise PcdError(
            "PCD input path must be a single .pcd file; directories are not "
            "supported: {}".format(path)
        )
    if path.suffix.lower() != ".pcd":
        raise PcdError("PCD input file must have a .pcd extension: {}".format(path))
    return path


def read_pcd_xyz(path):
    """Read the x/y/z fields from an ASCII, binary or binary_compressed PCD."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            header = _read_header(stream, path)
            if header["data"] == "ascii":
                points = _read_ascii_xyz(stream, header, path)
            elif header["data"] == "binary":
                points = _read_binary_xyz(stream, header, path)
            elif header["data"] == "binary_compressed":
                points = _read_binary_compressed_xyz(stream, header, path)
            else:
                raise PcdError(
                    "Unsupported PCD DATA type '{}' in {}".format(header["data"], path)
                )
    except OSError as error:
        raise PcdError("Could not read {}: {}".format(path, error)) from error

    if points.ndim != 2 or points.shape[1] != 3:
        raise PcdError("Invalid XYZ data shape in {}: {}".format(path, points.shape))
    return points


def voxel_downsample(xyz, voxel_size):
    """
    Keep one representative point per 3D voxel.

    Voxel coordinates are packed into one uint64 key so memory usage stays much
    lower than keeping an N x 3 int64 voxel-index array. The highest-Z source
    point in each voxel is retained so BEV height maxima are preserved.
    """
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise PcdError("voxel_size must be a finite value greater than zero")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise PcdError("Voxel input must have shape N x 3")

    finite = np.isfinite(xyz).all(axis=1)
    if not np.all(finite):
        xyz = xyz[finite]
    if xyz.shape[0] <= 1:
        return xyz

    max_uint64 = np.iinfo(np.uint64).max

    voxel_x = _voxel_indices(xyz[:, 0], voxel_size)
    min_voxel_x = int(np.min(voxel_x))
    size_x = int(np.max(voxel_x)) - min_voxel_x + 1
    keys = (voxel_x - min_voxel_x).astype(np.uint64)
    del voxel_x

    voxel_y = _voxel_indices(xyz[:, 1], voxel_size)
    min_voxel_y = int(np.min(voxel_y))
    size_y = int(np.max(voxel_y)) - min_voxel_y + 1
    if size_x > max_uint64 // size_y:
        raise PcdError("Voxel index range is too large; increase voxel_size")
    keys += (voxel_y - min_voxel_y).astype(np.uint64) * np.uint64(size_x)
    del voxel_y

    xy_size = size_x * size_y
    voxel_z = _voxel_indices(xyz[:, 2], voxel_size)
    min_voxel_z = int(np.min(voxel_z))
    size_z = int(np.max(voxel_z)) - min_voxel_z + 1
    if xy_size > max_uint64 // size_z:
        raise PcdError("Voxel index range is too large; increase voxel_size")
    keys += (voxel_z - min_voxel_z).astype(np.uint64) * np.uint64(xy_size)
    del voxel_z

    # Sort primarily by voxel key and secondarily by descending Z. The first
    # entry in each key group is therefore that voxel's highest source point.
    order = np.lexsort((-xyz[:, 2].astype(np.float64, copy=False), keys))
    sorted_keys = keys[order]
    first_in_voxel = np.empty(sorted_keys.shape[0], dtype=bool)
    first_in_voxel[0] = True
    first_in_voxel[1:] = sorted_keys[1:] != sorted_keys[:-1]
    representative_indices = order[first_in_voxel]
    if representative_indices.size == xyz.shape[0]:
        return xyz
    return xyz[representative_indices]


def load_point_cloud(pcd_file, voxel_size):
    """Read one complete PCD file, then voxel-filter its XYZ points once."""
    path = validate_pcd_file(pcd_file)
    xyz = read_pcd_xyz(path)
    total_points = int(xyz.shape[0])
    finite_points = int(np.count_nonzero(np.isfinite(xyz).all(axis=1)))
    if finite_points == 0:
        raise PcdError("The selected PCD file contains no finite XYZ points")

    sampled_xyz = voxel_downsample(xyz, voxel_size)
    del xyz
    sampled_xyz = np.ascontiguousarray(sampled_xyz, dtype=np.float32)
    mins = np.min(sampled_xyz, axis=0)
    maxs = np.max(sampled_xyz, axis=0)
    display_z_min, display_z_max = np.percentile(
        sampled_xyz[:, 2],
        DISPLAY_Z_PERCENTILES,
    )
    stats = {
        "file_count": 1,
        "total_points": total_points,
        "finite_points": finite_points,
        "sampled_points": int(sampled_xyz.shape[0]),
        "cache_bytes": int(sampled_xyz.nbytes),
        "voxel_size": float(voxel_size),
        "display_z_percentiles": DISPLAY_Z_PERCENTILES,
        "display_z_range": (float(display_z_min), float(display_z_max)),
        "xyz_bounds": (
            (float(mins[0]), float(maxs[0])),
            (float(mins[1]), float(maxs[1])),
            (float(mins[2]), float(maxs[2])),
        ),
    }
    return sampled_xyz, stats


def extent_for_bounds(
    x_bounds,
    y_bounds,
    aspect_ratio=16.0 / 9.0,
    padding_ratio=0.02,
):
    """
    Return a centered XY extent with the requested width/height aspect ratio.

    Matching the world extent aspect ratio to the raster aspect ratio keeps X
    and Y at the same physical resolution.
    """
    min_x, max_x = sorted((float(x_bounds[0]), float(x_bounds[1])))
    min_y, max_y = sorted((float(y_bounds[0]), float(y_bounds[1])))
    values = (min_x, max_x, min_y, max_y, aspect_ratio, padding_ratio)
    if not all(math.isfinite(value) for value in values):
        raise PcdError("BEV bounds, aspect ratio and padding must be finite")
    if aspect_ratio <= 0.0:
        raise PcdError("BEV aspect_ratio must be greater than zero")
    if padding_ratio < 0.0:
        raise PcdError("BEV padding_ratio must not be negative")

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    source_width = max_x - min_x
    source_height = max_y - min_y
    if source_width <= 0.0 and source_height <= 0.0:
        # A cloud with one XY location still needs a useful non-zero view.
        view_height = 1.0
    else:
        view_height = max(source_height, source_width / aspect_ratio)
    view_width = view_height * aspect_ratio
    scale = 1.0 + 2.0 * padding_ratio
    half_width = view_width * scale / 2.0
    half_height = view_height * scale / 2.0
    return (
        center_x - half_width,
        center_x + half_width,
        center_y - half_height,
        center_y + half_height,
    )


def build_bev(
    sampled_xyz,
    extent,
    grid_width=2560,
    grid_height=1440,
    dataset_stats=None,
):
    """
    Rasterize cached sampled XYZ points into a fixed grid for the requested extent.

    Each occupied pixel stores the maximum Z value of all points falling into it.
    """
    if sampled_xyz.ndim != 2 or sampled_xyz.shape[1] != 3:
        raise PcdError("Cached sampled point cloud must have shape N x 3")
    if sampled_xyz.shape[0] == 0:
        raise PcdError("Cached sampled point cloud is empty")
    if dataset_stats is None:
        raise PcdError("dataset_stats are required to build a BEV")
    if (
        isinstance(grid_width, bool)
        or not isinstance(grid_width, int)
        or grid_width <= 0
        or isinstance(grid_height, bool)
        or not isinstance(grid_height, int)
        or grid_height <= 0
    ):
        raise PcdError("BEV grid width and height must be positive integers")
    if len(extent) != 4:
        raise PcdError("BEV extent must contain min_x, max_x, min_y and max_y")

    min_x, max_x, min_y, max_y = (float(value) for value in extent)
    if not all(math.isfinite(value) for value in (min_x, max_x, min_y, max_y)):
        raise PcdError("BEV extent values must be finite")
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 0.0 or span_y <= 0.0:
        raise PcdError("BEV extent must have positive X and Y spans")
    extent_aspect = span_x / span_y
    grid_aspect = float(grid_width) / float(grid_height)
    if not math.isclose(
        extent_aspect,
        grid_aspect,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise PcdError(
            "BEV extent aspect ratio must match the raster aspect ratio"
        )

    resolution = span_x / float(grid_width)

    # Points on the maximum boundary are included in the last row/column.
    inside = (
        (sampled_xyz[:, 0] >= min_x)
        & (sampled_xyz[:, 0] <= max_x)
        & (sampled_xyz[:, 1] >= min_y)
        & (sampled_xyz[:, 1] <= max_y)
    )
    xyz = sampled_xyz[inside]
    visible_points = int(xyz.shape[0])
    raster_workers = min(BEV_RASTER_WORKERS, visible_points)
    if raster_workers:
        chunks = np.array_split(xyz, raster_workers)
        with ThreadPoolExecutor(
            max_workers=raster_workers,
            thread_name_prefix="bev-raster",
        ) as executor:
            futures = [
                executor.submit(
                    _rasterize_bev_chunk,
                    chunk,
                    min_x,
                    min_y,
                    resolution,
                    grid_width,
                    grid_height,
                )
                for chunk in chunks
            ]
            partial_grids = [future.result() for future in futures]

        # Each thread writes only to its own grid. Merge the four maximum-Z
        # grids in place so rasterization has no shared-write data race.
        grid = partial_grids[0]
        for partial_grid in partial_grids[1:]:
            np.maximum(grid, partial_grid, out=grid)
    else:
        grid = np.full(
            (grid_height, grid_width),
            -np.inf,
            dtype=np.float32,
        )

    occupied_pixels = int(np.count_nonzero(np.isfinite(grid)))
    return {
        "grid": grid,
        "extent": (min_x, max_x, min_y, max_y),
        "grid_width": grid_width,
        "grid_height": grid_height,
        "resolution": resolution,
        "file_count": dataset_stats["file_count"],
        "total_points": dataset_stats["total_points"],
        "finite_points": dataset_stats["finite_points"],
        "sampled_points": dataset_stats["sampled_points"],
        "cache_bytes": dataset_stats["cache_bytes"],
        "voxel_size": dataset_stats["voxel_size"],
        "display_z_percentiles": dataset_stats["display_z_percentiles"],
        "display_z_range": dataset_stats["display_z_range"],
        "visible_points": visible_points,
        "occupied_pixels": occupied_pixels,
        "raster_workers": raster_workers,
        "xyz_bounds": dataset_stats["xyz_bounds"],
    }


def _rasterize_bev_chunk(
    xyz,
    min_x,
    min_y,
    resolution,
    grid_width,
    grid_height,
):
    """Rasterize one point chunk into a thread-local maximum-Z grid."""
    grid = np.full((grid_height, grid_width), -np.inf, dtype=np.float32)
    if not xyz.size:
        return grid

    pixel_x = np.floor((xyz[:, 0] - min_x) / resolution).astype(np.int64)
    pixel_y = np.floor((xyz[:, 1] - min_y) / resolution).astype(np.int64)
    pixel_x = np.clip(pixel_x, 0, grid_width - 1)
    pixel_y = np.clip(pixel_y, 0, grid_height - 1)
    np.maximum.at(
        grid,
        (pixel_y, pixel_x),
        xyz[:, 2].astype(np.float32, copy=False),
    )
    return grid


def _voxel_indices(values, voxel_size):
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.floor(values / voxel_size)
    if not np.isfinite(scaled).all():
        raise PcdError("Voxel coordinates overflowed; increase voxel_size")
    int64_info = np.iinfo(np.int64)
    if np.min(scaled) < int64_info.min or np.max(scaled) > int64_info.max:
        raise PcdError("Voxel coordinates exceed int64 range; increase voxel_size")
    return scaled.astype(np.int64)


def _read_header(stream, path):
    raw_header = {}
    while True:
        line = stream.readline()
        if not line:
            raise PcdError("PCD header has no DATA line: {}".format(path))
        try:
            text = line.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise PcdError("PCD header is not ASCII: {}".format(path)) from error
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        key = parts[0].upper()
        values = parts[1:]
        raw_header[key] = values
        if key == "DATA":
            break

    required = ("FIELDS", "SIZE", "TYPE", "DATA")
    missing = [key for key in required if key not in raw_header]
    if missing:
        raise PcdError(
            "Missing PCD header field(s) {} in {}".format(", ".join(missing), path)
        )

    fields = [value.lower() for value in raw_header["FIELDS"]]
    try:
        sizes = [int(value) for value in raw_header["SIZE"]]
        types = [value.upper() for value in raw_header["TYPE"]]
        counts = [int(value) for value in raw_header.get("COUNT", ["1"] * len(fields))]
        if "POINTS" in raw_header:
            points = int(raw_header["POINTS"][0])
        else:
            width = int(raw_header.get("WIDTH", ["0"])[0])
            height = int(raw_header.get("HEIGHT", ["1"])[0])
            points = width * height
    except (ValueError, IndexError) as error:
        raise PcdError("Invalid numeric value in PCD header: {}".format(path)) from error

    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise PcdError("FIELDS/SIZE/TYPE/COUNT lengths differ in {}".format(path))
    if not raw_header["DATA"]:
        raise PcdError("PCD DATA field has no storage type: {}".format(path))
    if not all(axis in fields for axis in ("x", "y", "z")):
        raise PcdError("PCD has no complete x/y/z fields: {}".format(path))
    if any(count <= 0 for count in counts):
        raise PcdError("PCD COUNT values must be positive: {}".format(path))
    if points < 0:
        raise PcdError("PCD POINTS must not be negative: {}".format(path))

    return {
        "fields": fields,
        "sizes": sizes,
        "types": types,
        "counts": counts,
        "points": points,
        "data": raw_header["DATA"][0].lower(),
    }


def _read_ascii_xyz(stream, header, path):
    xyz_columns = _ascii_xyz_columns(header)
    try:
        points = np.loadtxt(
            stream,
            dtype=np.float32,
            comments="#",
            usecols=xyz_columns,
            ndmin=2,
        )
    except (ValueError, IndexError) as error:
        raise PcdError("Could not parse ASCII point data in {}: {}".format(path, error)) from error

    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return points


def _ascii_xyz_columns(header):
    field_columns = {}
    column = 0
    for name, count in zip(header["fields"], header["counts"]):
        field_columns[name] = column
        column += count
    return tuple(field_columns[axis] for axis in ("x", "y", "z"))


def _read_binary_xyz(stream, header, path):
    point_dtype, safe_names = _binary_layout(header, path)
    expected_size = header["points"] * point_dtype.itemsize
    data = stream.read(expected_size)
    if len(data) != expected_size:
        raise PcdError(
            "Binary payload in {} is {} bytes; expected {}".format(
                path, len(data), expected_size
            )
        )

    records = np.frombuffer(data, dtype=point_dtype, count=header["points"])
    return _binary_records_xyz(records, header, safe_names)


def _binary_layout(header, path):
    dtype_fields = []
    safe_names = []
    for index, (name, size, data_type, count) in enumerate(
        zip(
            header["fields"],
            header["sizes"],
            header["types"],
            header["counts"],
        )
    ):
        safe_name = "field_{}_{}".format(index, name)
        safe_names.append(safe_name)
        scalar_dtype = _pcd_dtype(data_type, size, path)
        if count == 1:
            dtype_fields.append((safe_name, scalar_dtype))
        else:
            dtype_fields.append((safe_name, scalar_dtype, (count,)))
    return np.dtype(dtype_fields), safe_names


def _binary_records_xyz(records, header, safe_names):
    columns = []
    for axis in ("x", "y", "z"):
        field_index = header["fields"].index(axis)
        values = records[safe_names[field_index]]
        if values.ndim > 1:
            values = values[:, 0]
        columns.append(values)
    return np.column_stack(columns)


def _read_binary_compressed_xyz(stream, header, path):
    sizes = stream.read(8)
    if len(sizes) != 8:
        raise PcdError("Missing binary_compressed size prefix in {}".format(path))
    compressed_size, uncompressed_size = struct.unpack("<II", sizes)
    compressed = stream.read(compressed_size)
    if len(compressed) != compressed_size:
        raise PcdError(
            "Compressed payload in {} is {} bytes; expected {}".format(
                path, len(compressed), compressed_size
            )
        )

    raw = _lzf_decompress(compressed, uncompressed_size, path)
    columns = {}
    offset = 0
    for name, size, data_type, count in zip(
        header["fields"],
        header["sizes"],
        header["types"],
        header["counts"],
    ):
        scalar_dtype = _pcd_dtype(data_type, size, path)
        value_count = header["points"] * count
        block_size = value_count * size
        if offset + block_size > len(raw):
            raise PcdError("binary_compressed field data is truncated in {}".format(path))
        if name in ("x", "y", "z"):
            values = np.frombuffer(
                raw,
                dtype=scalar_dtype,
                count=value_count,
                offset=offset,
            )
            if count > 1:
                values = values.reshape(header["points"], count)[:, 0]
            columns[name] = values
        offset += block_size

    return np.column_stack((columns["x"], columns["y"], columns["z"]))


def _pcd_dtype(data_type, size, path):
    dtype_codes = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "<i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "<u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    code = dtype_codes.get((data_type, size))
    if code is None:
        raise PcdError(
            "Unsupported PCD TYPE/SIZE combination {}/{} in {}".format(
                data_type, size, path
            )
        )
    return np.dtype(code)


def _lzf_decompress(compressed, expected_size, path):
    """Small LZF decoder used by PCD's binary_compressed storage mode."""
    output = bytearray()
    index = 0
    try:
        while index < len(compressed):
            control = compressed[index]
            index += 1
            if control < 32:
                length = control + 1
                output.extend(compressed[index:index + length])
                index += length
                continue

            length = control >> 5
            reference = len(output) - ((control & 0x1F) << 8) - 1
            if length == 7:
                length += compressed[index]
                index += 1
            reference -= compressed[index]
            index += 1
            length += 2
            if reference < 0:
                raise ValueError("invalid back-reference")
            for _ in range(length):
                output.append(output[reference])
                reference += 1
    except (IndexError, ValueError) as error:
        raise PcdError("Invalid LZF payload in {}".format(path)) from error

    if len(output) != expected_size:
        raise PcdError(
            "LZF payload in {} expands to {} bytes; expected {}".format(
                path, len(output), expected_size
            )
        )
    return bytes(output)
