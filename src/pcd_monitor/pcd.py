import math
import struct
from pathlib import Path

import numpy as np


class PcdError(RuntimeError):
    pass


def find_pcd_files(directory, recursive=False):
    """Return deterministically sorted PCD files under directory."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise PcdError("PCD directory does not exist or is not a directory: {}".format(root))

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".pcd"
    ]
    files.sort(key=lambda path: str(path).lower())
    if not files:
        raise PcdError("No .pcd files found in: {}".format(root))
    return files


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


def build_bev(pcd_files, resolution, max_grid_cells=16000000):
    """
    Rasterize all PCD files into one XY grid.

    Each occupied pixel stores the maximum Z value of all points falling into it.
    The return value is a dictionary containing the grid, its world extent and
    useful statistics for the UI.
    """
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise PcdError("BEV resolution must be a finite value greater than zero")
    if max_grid_cells <= 0:
        raise PcdError("max_grid_cells must be greater than zero")

    files = [Path(path) for path in pcd_files]
    if not files:
        raise PcdError("No PCD files were provided")

    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf
    total_points = 0
    finite_points = 0

    # The first pass finds world bounds without retaining every cloud in memory.
    for path in files:
        xyz = read_pcd_xyz(path)
        total_points += int(xyz.shape[0])
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        finite_points += int(xyz.shape[0])
        if xyz.size == 0:
            continue
        mins = np.min(xyz, axis=0)
        maxs = np.max(xyz, axis=0)
        min_x = min(min_x, float(mins[0]))
        min_y = min(min_y, float(mins[1]))
        min_z = min(min_z, float(mins[2]))
        max_x = max(max_x, float(maxs[0]))
        max_y = max(max_y, float(maxs[1]))
        max_z = max(max_z, float(maxs[2]))

    if finite_points == 0:
        raise PcdError("The selected PCD files contain no finite XYZ points")

    origin_x = math.floor(min_x / resolution) * resolution
    origin_y = math.floor(min_y / resolution) * resolution
    width = int(math.floor((max_x - origin_x) / resolution)) + 1
    height = int(math.floor((max_y - origin_y) / resolution)) + 1
    cell_count = width * height
    if cell_count > max_grid_cells:
        approximate_resolution = resolution * math.sqrt(
            float(cell_count) / float(max_grid_cells)
        )
        raise PcdError(
            "BEV grid would contain {:,} pixels ({} x {}), above the {:,} limit. "
            "Increase ~resolution to about {:.4g} m/pixel or raise "
            "~max_grid_cells.".format(
                cell_count,
                width,
                height,
                max_grid_cells,
                approximate_resolution,
            )
        )

    grid = np.full((height, width), -np.inf, dtype=np.float32)

    # The second pass performs an in-place maximum reduction for each pixel.
    for path in files:
        xyz = read_pcd_xyz(path)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.size == 0:
            continue

        pixel_x = np.floor((xyz[:, 0] - origin_x) / resolution).astype(np.int64)
        pixel_y = np.floor((xyz[:, 1] - origin_y) / resolution).astype(np.int64)
        inside = (
            (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
        )
        np.maximum.at(
            grid,
            (pixel_y[inside], pixel_x[inside]),
            xyz[inside, 2].astype(np.float32, copy=False),
        )

    occupied = np.isfinite(grid)
    extent = (
        origin_x,
        origin_x + width * resolution,
        origin_y,
        origin_y + height * resolution,
    )
    return {
        "grid": grid,
        "occupied": occupied,
        "extent": extent,
        "resolution": resolution,
        "file_count": len(files),
        "total_points": total_points,
        "finite_points": finite_points,
        "occupied_pixels": int(np.count_nonzero(occupied)),
        "xyz_bounds": ((min_x, max_x), (min_y, max_y), (min_z, max_z)),
    }


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
    field_columns = {}
    column = 0
    for name, count in zip(header["fields"], header["counts"]):
        field_columns[name] = column
        column += count
    xyz_columns = tuple(field_columns[axis] for axis in ("x", "y", "z"))

    try:
        points = np.loadtxt(
            stream,
            dtype=np.float64,
            comments="#",
            usecols=xyz_columns,
            ndmin=2,
        )
    except (ValueError, IndexError) as error:
        raise PcdError("Could not parse ASCII point data in {}: {}".format(path, error)) from error

    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return points


def _read_binary_xyz(stream, header, path):
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

    point_dtype = np.dtype(dtype_fields)
    expected_size = header["points"] * point_dtype.itemsize
    data = stream.read(expected_size)
    if len(data) != expected_size:
        raise PcdError(
            "Binary payload in {} is {} bytes; expected {}".format(
                path, len(data), expected_size
            )
        )

    records = np.frombuffer(data, dtype=point_dtype, count=header["points"])
    columns = []
    for axis in ("x", "y", "z"):
        field_index = header["fields"].index(axis)
        values = records[safe_names[field_index]]
        if values.ndim > 1:
            values = values[:, 0]
        columns.append(values.astype(np.float64, copy=False))
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
            columns[name] = values.astype(np.float64, copy=False)
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
