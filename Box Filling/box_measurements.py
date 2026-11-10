import numpy as np


def get_top_edge_point(x_frame_array, y_frame_array, z_frame_array, u, v, size=10, tolerance=30):
    """
    Get the real-world (X, Y, Z) point of the box's top edge near pixel (u, v).
    As vertices may also have ground information or incorrect info which is not the top
    edge, a window of +/- size is kept. Pixels within `tolerance` mm of the minimum
    (non-zero) depth in it are the top edge, and the one closest to the vertex is taken
    as the box's corner.
    """
    rows, cols = z_frame_array.shape
    # Clamp the window to the frame so negative indices don't wrap around
    v0, v1 = max(v - size, 0), min(v + size, rows)
    u0, u1 = max(u - size, 0), min(u + size, cols)

    z_window = z_frame_array[v0:v1, u0:u1]
    valid = z_window > 0
    if not np.any(valid):
        return None

    # Top edge pixels: close to the minimum non-zero depth inside the window
    min_z = np.min(z_window[valid])
    rr, cc = np.nonzero(valid & (z_window <= min_z + tolerance))

    # Corner: the top edge pixel closest to the vertex
    nearest = np.argmin((rr + v0 - v) ** 2 + (cc + u0 - u) ** 2)
    r, c = rr[nearest], cc[nearest]

    return (x_frame_array[v0 + r, u0 + c],
            y_frame_array[v0 + r, u0 + c],
            z_frame_array[v0 + r, u0 + c])


def calculate_box_dimensions(approx, z_frame_array, x_frame_array, y_frame_array):
    """
    Calculate the width and length of a box in mm.
    The sensor already provides X, Y, Z frames in real-world mm, so the distances between
    the box's top corners are measured directly from them (no intrinsics or scale factor needed).

    Returns:
    (width, length) in mm, or (None, None) if the corners can't be measured.
    """
    if len(approx) != 4:
        return None, None

    corners = []
    for i in range(4):
        u, v = (int(p) for p in approx[i][0])
        point = get_top_edge_point(x_frame_array, y_frame_array, z_frame_array, u, v)
        if point is None:
            return None, None
        corners.append(np.array(point, dtype=np.float64))

    # Distances between consecutive corners
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    sides = sorted(sides)

    width = np.average(sides[:2])   # Smaller sides are the width
    length = np.average(sides[2:])  # Larger sides are the length

    return width, length
