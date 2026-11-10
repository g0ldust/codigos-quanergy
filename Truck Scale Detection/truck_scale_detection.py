import warnings
import cv2
import numpy as np
from collections import deque
from datetime import datetime

HEIGHT_THRESHOLD_MM = 50     # A point counts as "truck" if it is this high (mm) above the floor plane
CHANGE_THRESHOLD_MM = 50     # ...and its depth changed this much (mm) from the empty reference (ignores furniture)
OUTSIDE_TOLERANCE_MM = 10    # A truck point counts as outside only if it is this far (mm) past the zone edge
OUTSIDE_MIN_PIXELS = 20      # Points outside needed to consider the truck outside (avoids noise)
MIN_OBJECT_PIXELS = 300      # Smaller objects touching the zone are ignored (noise, a hand, etc.)
TEMPORAL_THRESHOLD = 5       # Number of consecutive frames to confirm a change of state
REFERENCE_FRAMES = 10        # Frames combined to capture the reference of the empty zone
CAMERA_MODE = 5              # Medium Range Static Scene
IMAGE_ROTATION = 0           # Extra 90° clockwise turns of the image (2 = sensor mounted upside down). Key 'r' changes it
SMOOTHING_FRAMES = 5         # Frames averaged per pixel to reduce flicker (1 = off)
MEDIAN_FILTER_SIZE = 5       # Median filter to remove isolated noisy pixels (3 or 5, 0 = off)

# States of the scale
EMPTY, INSIDE, OUTSIDE = "empty", "inside", "outside"


def get_frame_arrays(frames, frame_types, rotation=0):
    '''
    Frames as contiguous float arrays rotated to the sensor's display orientation, in the order of `frame_types`
    (X, Y, Z in mm, then intensity), plus `rotation` extra 90° clockwise turns. Only the image turns: X, Y, Z
    values stay the same. Frames are matched by their type: the camera doesn't always return them in the order
    they were requested. Each frame is read in its own data type and converted afterwards: the SDK can't
    convert it with NumPy 2.
    '''
    by_type = {int(f.frameType()): f for f in frames}
    return [np.ascontiguousarray(np.rot90(np.asarray(by_type[int(t)]).astype(np.float32), -(1 + rotation)))
            for t in frame_types]


def smooth_xyz(x, y, z, history):
    '''
    Reduce noise: per-pixel average of the last frames (only valid readings; a pixel valid in less than
    half of them is set invalid, which removes flicker), then a median filter on the depth for isolated
    noisy pixels. X and Y are scaled with the filtered depth so each point stays on its pixel's ray.
    '''
    history.append((x, y, z))
    if len(history) > 1:
        valid = np.stack([h[2] for h in history]) > 0
        count = np.count_nonzero(valid, axis=0)
        keep = count >= (len(history) + 1) // 2
        x, y, z = [np.where(keep, (np.stack([h[k] for h in history]) * valid).sum(axis=0) / np.maximum(count, 1), 0)
                   .astype(np.float32) for k in range(3)]
    if MEDIAN_FILTER_SIZE in (3, 5):
        z_median = cv2.medianBlur(z, MEDIAN_FILTER_SIZE)
        # Holes the median fills have no X, Y: they stay invalid
        scale = np.where(z > 0, z_median / np.maximum(z, 1e-6), 0).astype(np.float32)
        x, y, z = x * scale, y * scale, z * scale
    return x, y, z


def depth_to_color(depth):
    '''
    Colored depth for display. The color range ignores invalid pixels and the 1% extremes, so it doesn't
    jump from frame to frame. Invalid pixels (no reading) are shown in black.
    '''
    valid = depth > 0
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)
    low, high = np.percentile(depth[valid], [1, 99])
    normalized = np.clip((depth - low) / max(high - low, 1), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def capture_reference(frames_list):
    '''
    Reference of the empty zone: per-pixel median of X, Y and Z over several frames,
    ignoring invalid (zero depth) readings. Pixels with no valid reading are left at 0.
    '''
    reference = []
    valid = np.stack([f[2] for f in frames_list]) > 0
    for k in range(3):
        stack = np.stack([f[k] for f in frames_list]).astype(np.float32)
        stack[~valid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reference.append(np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0))
    return reference  # [x, y, z]


def fit_floor_plane(x, y, z, mask):
    '''
    Fit the floor plane to the reference points inside the mask (the empty zone).
    Returns (origin, normal, u, v): normal points to the sensor, u and v are horizontal axes on the floor.
    With this the sensor doesn't need to look perfectly straight down.
    '''
    sel = mask & (z > 0)
    points = np.column_stack([x[sel], y[sel], z[sel]]).astype(np.float64)
    if len(points) < 100:
        return None

    # Least squares plane, removing outliers (tape bumps, noise) in a few iterations
    for _ in range(3):
        origin = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - origin, full_matrices=False)
        normal = vt[2]
        dist = np.abs((points - origin) @ normal)
        keep = dist < max(3 * dist.std(), 10)
        if keep.sum() < 100:
            break
        points = points[keep]
    origin = points.mean(axis=0)

    # The sensor is at (0, 0, 0): make the normal point towards it, so height above the floor is positive
    if np.dot(-origin, normal) < 0:
        normal = -normal
    u = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return origin, normal, u, v


def to_floor_coords(x, y, z, plane):
    ''' Height above the floor (h) and horizontal position on the floor (u, v), all in mm '''
    origin, normal, u_axis, v_axis = plane
    dx, dy, dz = x - origin[0], y - origin[1], z - origin[2]
    h = dx * normal[0] + dy * normal[1] + dz * normal[2]
    u = dx * u_axis[0] + dy * u_axis[1] + dz * u_axis[2]
    v = dx * v_axis[0] + dy * v_axis[1] + dz * v_axis[2]
    return h, u, v


def pixel_to_floor(reference, plane, px, py, size=3):
    ''' Floor (u, v) of an image pixel, as the median of the valid reference points in a small window '''
    x_ref, y_ref, z_ref = reference
    rows, cols = z_ref.shape
    y0, y1 = max(py - size, 0), min(py + size + 1, rows)
    x0, x1 = max(px - size, 0), min(px + size + 1, cols)
    valid = z_ref[y0:y1, x0:x1] > 0
    if not np.any(valid):
        return None
    point = [np.median(a[y0:y1, x0:x1][valid]) for a in reference]
    _, u, v = to_floor_coords(*point, plane)
    return float(u), float(v)


def points_in_polygon(pu, pv, polygon):
    ''' Vectorized ray casting: True for the points (pu, pv) inside the polygon [(u, v), ...] '''
    inside = np.zeros(pu.shape, dtype=bool)
    n = len(polygon)
    for i in range(n):
        u1, v1 = polygon[i]
        u2, v2 = polygon[(i + 1) % n]
        crosses = (v1 > pv) != (v2 > pv)
        u_cross = (u2 - u1) * (pv - v1) / (v2 - v1 + 1e-12) + u1
        inside ^= crosses & (pu < u_cross)
    return inside


def distance_to_polygon(pu, pv, polygon):
    ''' Distance (mm) from each point to the closest edge of the polygon '''
    best = np.full(pu.shape, np.inf)
    n = len(polygon)
    for i in range(n):
        a = np.array(polygon[i])
        b = np.array(polygon[(i + 1) % n])
        ab = b - a
        t = np.clip(((pu - a[0]) * ab[0] + (pv - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.hypot(pu - (a[0] + t * ab[0]), pv - (a[1] + t * ab[1])))
    return best


def check_truck(x, y, z, reference, plane, zone):
    '''
    Find the truck on the zone and check whether it is completely inside.
    Returns (state, truck_mask, outside_mask, max_outside_mm).
    '''
    h, u, v = to_floor_coords(x, y, z, plane)
    z_ref = reference[2]
    valid = (z > 0) & (z_ref > 0)

    # Truck points: above the floor and different from the empty reference
    obj = valid & (h > HEIGHT_THRESHOLD_MM) & (np.abs(z - z_ref) > CHANGE_THRESHOLD_MM)
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Removes edge noise

    rows, cols = np.nonzero(obj)
    inside = points_in_polygon(u[rows, cols], v[rows, cols], zone)

    # Only objects touching the zone are the truck; a person walking around elsewhere is ignored
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    labels_in_zone = np.unique(labels[rows[inside], cols[inside]])
    labels_in_zone = [l for l in labels_in_zone if l != 0 and stats[l, cv2.CC_STAT_AREA] >= MIN_OBJECT_PIXELS]
    truck = np.isin(labels[rows, cols], labels_in_zone)

    truck_mask = np.zeros(z.shape, dtype=bool)
    outside_mask = np.zeros(z.shape, dtype=bool)
    if not np.any(truck):
        return EMPTY, truck_mask, outside_mask, 0.0
    truck_mask[rows[truck], cols[truck]] = True

    # Truck points outside the zone, beyond the tolerance
    out = truck & ~inside
    dist = distance_to_polygon(u[rows[out], cols[out]], v[rows[out], cols[out]], zone)
    beyond = dist > OUTSIDE_TOLERANCE_MM
    outside_mask[rows[out][beyond], cols[out][beyond]] = True
    max_outside = float(dist.max()) if dist.size > 0 else 0.0

    state = OUTSIDE if beyond.sum() >= OUTSIDE_MIN_PIXELS else INSIDE
    return state, truck_mask, outside_mask, max_outside


# Function to highlight text with background
def highlight_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, text_color=(0, 0, 255), highlight_color=(0, 255, 255)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position

    # Draw a filled rectangle behind the text (highlight color)
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), highlight_color, thickness=cv2.FILLED)

    # Draw the text on top of the rectangle (text color)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)


# Corners of the scale (tape) clicked with the mouse, in image pixels
corners = []


def click_event(event, x, y, flags, param):
    ''' Left click: add a corner of the scale (4 in total). Right click: clear the corners. '''
    global corners
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        corners = []


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


gpio_enabled = True  # False if the camera doesn't accept the GPIO configuration


def set_outputs(cam, state):
    ''' OUT1 goes low (0) while the truck is outside the scale, OUT2 goes low (0) while it is completely inside '''
    if not gpio_enabled:
        return
    cam.setGpio(1, 0 if state == OUTSIDE else 1)
    cam.setGpio(2, 0 if state == INSIDE else 1)


if __name__ == "__main__":
    import chronoptics.tof as tof

    # Initialize 3D camera
    serial = input("Enter sensor's serial number: ")
    cam = tof.TuiCamera(serial=serial)

    # X, Y, Z for measuring; intensity (if available) to see the tape on the floor
    frame_types = [tof.FrameType.X, tof.FrameType.Y, tof.FrameType.Z]
    intensity_type = getattr(tof.FrameType, "INTENSITY", None)
    if intensity_type is not None:
        frame_types.append(intensity_type)
    cam.setOutputFrameTypes(frame_types)

    # Continuous is the camera's default trigger mode: if the camera doesn't accept the setting, keep going
    try:
        cam.setTriggerMode(tof.TriggerMode.CONTINUOUS)
    except RuntimeError as e:
        log(f"Warning: could not set the trigger mode ({e}), using the camera's default.")

    # Configure GPIO: OUT1 = truck outside, OUT2 = truck inside (both active low, like the other applets)
    try:
        cam.setGpioMode(0, tof.GpioMode.GENERAL_INPUT)
        cam.setGpioMode(1, tof.GpioMode.GENERAL_OUTPUT)
        cam.setGpioMode(2, tof.GpioMode.GENERAL_OUTPUT)
        # Initializing output pins to idle (high) state
        cam.setGpio(1, 1)
        cam.setGpio(2, 1)
    except RuntimeError as e:
        gpio_enabled = False
        log(f"Warning: could not configure GPIO ({e}). Running without outputs, only on screen.")

    cam.start()
    cam.switchConfig(CAMERA_MODE)

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_event)

    reference = None             # [x, y, z] of the empty zone
    reference_buffer = []        # Frames collected while capturing the reference
    capturing_reference = False
    calibrated_corners = None    # Corners the floor plane and zone were computed with
    plane = None                 # Floor plane
    zone = None                  # Corners of the scale on the floor (u, v) in mm
    state = EMPTY                # Confirmed state
    candidate, candidate_count = EMPTY, 0
    show_intensity = intensity_type is not None
    rotation = IMAGE_ROTATION
    xyz_history = deque(maxlen=max(SMOOTHING_FRAMES, 1))  # Last frames, for the noise filter

    print("1. With the scale EMPTY, click its 4 corners (the tape) in order around it. Right click clears them.")
    print("2. Press 'f' to save the empty scale as reference.")
    print("Keys: f = save reference, n = reset, v = change view, r = rotate image 90°, q = quit.")

    while cam.isStreaming():
        frames = cam.getFrames()
        arrays = get_frame_arrays(frames, frame_types, rotation)
        x, y, z = smooth_xyz(*arrays[:3], xyz_history)

        # Image to display: intensity shows the tape, depth shows the heights
        if show_intensity:
            intensity = np.clip(arrays[3], 0, np.percentile(arrays[3], 99))
            gray = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            frame = depth_to_color(z)

        # Read the keyboard only once per frame so key presses aren't lost
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('v') and intensity_type is not None:
            show_intensity = not show_intensity

        # Rotating the image moves every pixel, so the corners and the reference must be taken again
        if key == ord('r'):
            rotation = (rotation + 1) % 4
            xyz_history.clear()  # Old frames have the previous orientation
            log(f"Image rotated {rotation * 90}°. To keep it, set IMAGE_ROTATION = {rotation} at the top of the file.")

        if key in (ord('n'), ord('r')):
            corners = []
            reference = None
            capturing_reference = False
            plane = zone = calibrated_corners = None
            state, candidate, candidate_count = EMPTY, EMPTY, 0
            set_outputs(cam, state)
            log("Reset. Click the 4 corners of the empty scale.")

        if key == ord('f'):
            if len(corners) < 4:
                log("First click the 4 corners of the scale.")
            else:
                reference_buffer = []
                capturing_reference = True
                log("Capturing reference, the scale must be empty...")

        if capturing_reference:
            reference_buffer.append([x, y, z])
            if len(reference_buffer) >= REFERENCE_FRAMES:
                reference = capture_reference(reference_buffer)
                capturing_reference = False
                calibrated_corners = None  # Forces the calibration below

        # Floor plane and zone, computed again if the corners or the reference change
        if reference is not None and len(corners) == 4 and corners != calibrated_corners:
            zone_mask = np.zeros(z.shape, dtype=np.uint8)
            cv2.fillPoly(zone_mask, [np.array(corners, dtype=np.int32)], 1)
            plane = fit_floor_plane(*reference, zone_mask.astype(bool))
            zone = None
            if plane is None:
                log("Warning: the floor has no valid readings inside the scale. Check the sensor and the corners.")
            else:
                zone = [pixel_to_floor(reference, plane, cx, cy) for cx, cy in corners]
                if any(c is None for c in zone):
                    zone = None
                    log("Warning: a corner has no valid reading. Click it again a bit inside the tape.")
                else:
                    sides = [np.hypot(*np.subtract(zone[i], zone[(i + 1) % 4])) for i in range(4)]
                    log("Scale measured: sides " + ", ".join(f"{s:.0f}" for s in sides) + " mm. Monitoring truck.")
            calibrated_corners = list(corners)
            state, candidate, candidate_count = EMPTY, EMPTY, 0
            set_outputs(cam, state)

        # Draw the scale's corners and edges
        for c in corners:
            cv2.circle(frame, c, 4, (0, 255, 255), -1)
        if len(corners) >= 2:
            cv2.polylines(frame, [np.array(corners, dtype=np.int32)], len(corners) == 4, (0, 255, 255), 2)

        if zone is None:
            if capturing_reference:
                message = "Guardando referencia..."
            elif len(corners) < 4:
                message = f"Marca las esquinas de la balanza ({len(corners)}/4)"
            else:
                message = "Con la balanza vacia, presiona 'f'"
            highlight_text(frame, message, (10, 30), font_scale=0.7, text_color=(0, 0, 0), highlight_color=(255, 255, 255))
        else:
            current, truck_mask, outside_mask, max_outside = check_truck(x, y, z, reference, plane, zone)

            # A change of state is confirmed after several consecutive frames
            if current == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count = current, 1
            if candidate != state and candidate_count >= TEMPORAL_THRESHOLD:
                state = candidate
                set_outputs(cam, state)
                log({EMPTY: "Scale empty", INSIDE: "Truck completely inside the scale",
                     OUTSIDE: f"TRUCK OUTSIDE THE SCALE ({max_outside:.0f} mm past the edge)"}[state])

            # Truck in green, the part outside the scale in red
            overlay = frame.copy()
            overlay[truck_mask] = (0, 200, 0)
            overlay[outside_mask] = (0, 0, 255)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            if state == OUTSIDE:
                highlight_text(frame, "CAMION FUERA DE LA BALANZA!", (10, 30), font_scale=1, text_color=(0, 0, 255), highlight_color=(0, 255, 255))
            elif state == INSIDE:
                highlight_text(frame, "Camion dentro: pesaje valido", (10, 30), font_scale=0.8, text_color=(0, 0, 0), highlight_color=(0, 255, 0))
            else:
                highlight_text(frame, "Balanza vacia", (10, 30), font_scale=0.8, text_color=(0, 0, 0), highlight_color=(255, 255, 255))

            if current == OUTSIDE:
                cv2.putText(frame, f"Fuera: {max_outside:.0f} mm (tolerancia {OUTSIDE_TOLERANCE_MM} mm)", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)

    set_outputs(cam, EMPTY)
    cam.stop()
    cv2.destroyAllWindows()
    cam = None
