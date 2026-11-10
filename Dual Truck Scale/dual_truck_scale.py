import gc
import multiprocessing as mp
import queue
import time
import warnings
import cv2
import numpy as np
from collections import deque
from datetime import datetime

SERIALS = []                 # Serial numbers of the sensors, e.g. ["3030031", "3030034"]. Empty = asked at start
IMAGE_ROTATIONS = [0, 0]     # Extra 90° clockwise turns of each sensor's image (2 = upside down). Key 'r' changes it

# Scale model: the 60 x 60 x 35 cm box is the truck. Truck 2.2 m wide -> scale 1 : 3.67
# (real truck: TRUCK_WIDTH_MM = 2200, WIDTH_TOLERANCE_MM = 200, WHEEL_BAND_MM = 1100, MAX_OUTSIDE_MM = 1000)
TRUCK_WIDTH_MM = 600         # Width of the truck (the box). The scale in the model should be 3.0 m / 3.67 = 818 mm wide
WIDTH_TOLERANCE_MM = 60      # The width measured by both sensors must be TRUCK_WIDTH_MM +- this, otherwise "doubtful"
MIN_MARGIN_MM = 0            # Minimum distance (mm) from the wheels to the edge of the scale on each side (0 = right at the edge)
WHEEL_BAND_MM = 300          # Only points up to this height (mm) count: the wheels, not the body or the hopper (real: ~1.1 m)
END_TOLERANCE_MM = 20        # Wheels past the ends of the scale (front / back) by more than this count as outside
HEIGHT_THRESHOLD_MM = 50     # A point counts as "truck" if it is this high (mm) above the floor
CHANGE_THRESHOLD_MM = 50     # ...and its depth changed this much (mm) from the empty reference (ignores furniture)
MAX_OUTSIDE_MM = 300         # Points farther than this (mm) from the scale are other things (a person nearby), ignored
OUTSIDE_MIN_PIXELS = 20      # Points past the ends needed to consider a wheel outside (avoids noise)
MIN_WHEEL_PIXELS = 150       # Points needed to consider that a sensor sees the truck
TEMPORAL_THRESHOLD = 5       # Number of consecutive frames to confirm a change of state
REFERENCE_FRAMES = 10        # Frames combined to capture the reference of the empty area
CAMERA_MODE = 5              # Medium Range Static Scene
SMOOTHING_FRAMES = 5         # Frames averaged per pixel to reduce flicker (1 = off)
MEDIAN_FILTER_SIZE = 5       # Median filter to remove isolated noisy pixels (3 or 5, 0 = off)
USE_GPIO = False             # True = use the outputs OUT1/OUT2. Off: configuring them broke the network connection of the sensors
RECONNECT_SECONDS = 5        # If a sensor loses the connection, try to reconnect it every this many seconds

# States of the scale
EMPTY, INSIDE, OUTSIDE, DOUBTFUL = "empty", "inside", "outside", "doubtful"
STATE_TEXT = {EMPTY: "Balanza vacia", INSIDE: "CAMION DENTRO", OUTSIDE: "RUEDA FUERA DE LA BALANZA!",
              DOUBTFUL: "Medicion dudosa"}


def frames_by_type(frames):
    ''' Frames indexed by their type: the camera doesn't always return them in the order they were requested '''
    return {int(f.frameType()): f for f in frames}


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


def color_to_image(color, shape):
    '''
    Projected color frame as an image (rows, cols, 3) aligned with the depth image, or None if its layout
    is unexpected. Pixels of the depth image that the color camera doesn't see come black.
    '''
    rows, cols = shape
    if color.size % (rows * cols) != 0:
        return None
    channels = color.size // (rows * cols)
    if channels not in (3, 4):
        return None
    return np.ascontiguousarray(color.reshape(rows, cols, channels)[..., :3]).astype(np.uint8)


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
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def intensity_to_gray(intensity):
    ''' Intensity (infrared image) as 0-255 gray, ignoring the brightest 1% (reflections) '''
    high = np.percentile(intensity, 99)
    return (np.clip(intensity / max(high, 1e-6), 0, 1) * 255).astype(np.uint8)


def render_view(view, depth, intensity):
    '''
    Image to display:
    - "combined": depth colors shaded with the intensity, shows the distance and also the details.
    - "depth": only the depth colors.
    - "intensity": the infrared image in gray, like a normal camera (shows the tape on the floor).
    '''
    if view == "intensity" and intensity is not None:
        return cv2.cvtColor(intensity_to_gray(intensity), cv2.COLOR_GRAY2BGR)
    colored = depth_to_color(depth)
    if view == "combined" and intensity is not None:
        shade = 0.25 + 0.75 * intensity_to_gray(intensity).astype(np.float32) / 255
        colored = (colored * shade[..., None]).astype(np.uint8)
    return colored


def capture_reference(frames_list):
    '''
    Reference of the empty area: per-pixel median of X, Y and Z over several frames,
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
    Fit the floor plane to the reference points inside the mask (the empty area).
    Returns (origin, normal, u, v): normal points to the sensor, u and v are horizontal axes on the floor.
    With this each sensor can look at the area from any angle.
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


def scale_side(zone, plane):
    '''
    The side of the scale this sensor measures: the edge in front of the sensor (it looks at the wheels of
    that side). Returns the edge's start point, its direction along the scale (d), the direction across the
    scale towards the inside (n), its index, its length and the width of the scale (distance to the opposite edge).
    '''
    _, su, sv = to_floor_coords(0.0, 0.0, 0.0, plane)  # The sensor is at (0, 0, 0): its position on the floor
    corners = np.array(zone, dtype=np.float64)
    center = corners.mean(axis=0)
    best = None
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        if (center - a) @ n < 0:
            n = -n
        # How far the sensor is outside this edge: the edge in front of the sensor is the one it is most outside of
        outside = (np.array([su, sv]) - a) @ -n
        if best is None or outside > best[0]:
            best = (outside, i, a, d, n, float(np.linalg.norm(b - a)))
    _, i, a, d, n, length = best
    width = float(np.mean([(corners[(i + 2) % 4] - a) @ n, (corners[(i + 3) % 4] - a) @ n]))
    return {"index": i, "a": a, "d": d, "n": n, "length": length, "width": width}


def check_wheels(x, y, z, reference, plane, side):
    '''
    Find the truck's wheels on this sensor's side using the depth and measure the margin: distance (mm) from the
    outer face of the wheels to the edge of the scale, along the floor (positive = inside, negative = outside).
    Points are placed on the floor with their real X, Y, Z, so the sensor's angle doesn't change the measure.
    Returns (result, wheel_mask, outside_mask); result is None if the sensor doesn't see the truck, otherwise
    {"margin": mm, "end_out": True if a wheel is past the front or back end of the scale}.
    '''
    h, u, v = to_floor_coords(x, y, z, plane)
    z_ref = reference[2]
    valid = (z > 0) & (z_ref > 0)

    # Truck points: above the floor, different from the empty reference, and low (wheels, not the body)
    obj = valid & (h > HEIGHT_THRESHOLD_MM) & (h < WHEEL_BAND_MM) & (np.abs(z - z_ref) > CHANGE_THRESHOLD_MM)
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Removes edge noise

    rows, cols = np.nonzero(obj)
    pu, pv = u[rows, cols] - side["a"][0], v[rows, cols] - side["a"][1]
    across = pu * side["n"][0] + pv * side["n"][1]   # From this side's edge towards the inside of the scale
    along = pu * side["d"][0] + pv * side["d"][1]    # Along the scale, from 0 to its length

    # Only points on the scale or close to it: farther away they are other things
    near = ((across > -MAX_OUTSIDE_MM) & (across < side["width"] + MAX_OUTSIDE_MM)
            & (along > -MAX_OUTSIDE_MM) & (along < side["length"] + MAX_OUTSIDE_MM))
    wheel_mask = np.zeros(z.shape, dtype=bool)
    outside_mask = np.zeros(z.shape, dtype=bool)
    if near.sum() < MIN_WHEEL_PIXELS:
        return None, wheel_mask, outside_mask
    rows, cols, across, along = rows[near], cols[near], across[near], along[near]
    wheel_mask[rows, cols] = True

    # Outer face of the wheels: the points closest to the edge (a low percentile, so a few noisy points don't count)
    margin = float(np.percentile(across, 2))
    past_end = (along < -END_TOLERANCE_MM) | (along > side["length"] + END_TOLERANCE_MM)
    out = (across < MIN_MARGIN_MM) | past_end
    outside_mask[rows[out], cols[out]] = True
    return {"margin": margin, "end_out": bool(past_end.sum() >= OUTSIDE_MIN_PIXELS)}, wheel_mask, outside_mask


def combine_results(sensors):
    '''
    Result of the scale from the sensors that are measuring (calibrated and connected), one on each side.
    - No sensor sees the truck: empty.
    - Both see it: the width of the truck is the width of the scale minus both margins. If it isn't the truck's
      width, the measure is not the truck (a person, a sensor partly covered...): doubtful.
    - Any wheel closer to the edge than MIN_MARGIN_MM, or past an end: outside. Otherwise: inside.
    Returns (state, width or None, numbers of the sensors that see a wheel outside).
    '''
    seen = [s for s in sensors if s.result is not None]
    if not seen:
        return EMPTY, None, []
    width = None
    if len(seen) == 2:
        scale_width = np.mean([s.side["width"] for s in seen])
        width = float(scale_width - seen[0].result["margin"] - seen[1].result["margin"])
        if abs(width - TRUCK_WIDTH_MM) > WIDTH_TOLERANCE_MM:
            return DOUBTFUL, width, []
    out = [s.index + 1 for s in seen if s.result["margin"] < MIN_MARGIN_MM or s.result["end_out"]]
    return (OUTSIDE if out else INSIDE), width, out


# Function to highlight text with background
def highlight_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, text_color=(0, 0, 255), highlight_color=(0, 255, 255)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position

    # Draw a filled rectangle behind the text (highlight color)
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), highlight_color, thickness=cv2.FILLED)

    # Draw the text on top of the rectangle (text color)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def sensor_process(serial, frames_out, messages_out, commands_in):
    '''
    Runs in its own process, one per sensor: connects to the sensor, takes its frames out all the time and
    sends the latest ones to the main program. With both sensors in the same process the SDK lost the
    connection of both after 30-60 s; each one in its own process streams without cuts.
    Messages to the main program: ("connected", None), ("lost", error), ("log", text).
    Commands from the main program: ("outputs", state), ("stop", None).
    '''
    import chronoptics.tof as tof
    frames_out.cancel_join_thread()  # On exit, don't wait for images the main program won't read anymore
    frame_types = [tof.FrameType.X, tof.FrameType.Y, tof.FrameType.Z, tof.FrameType.INTENSITY]
    outputs = EMPTY
    mode = CAMERA_MODE
    failures = 0

    while True:
        cam, gpio_enabled, step = None, False, "connecting"
        try:
            cam = tof.TuiCamera(serial=serial)
            # Color image projected onto the depth pixels (if the sensor has a color camera): shows the tape on the floor
            color_type = getattr(tof.FrameType, "BGR_PROJECTED", None)
            types = list(frame_types)
            try:
                if color_type is not None and int(color_type) in [int(t) for t in cam.possibleFrameTypes()]:
                    cam.setOutputFrameTypes(frame_types + [color_type])
                    types.append(color_type)
                else:
                    cam.setOutputFrameTypes(frame_types)
            except RuntimeError:
                cam.setOutputFrameTypes(frame_types)
                types = list(frame_types)
            # The trigger mode is not set: the camera's default (continuous) is used, setting it failed on these sensors

            # GPIO: OUT1 = a wheel outside the scale, OUT2 = truck inside (both active low, like the other applets)
            if USE_GPIO:
                try:
                    cam.setGpioMode(0, tof.GpioMode.GENERAL_INPUT)
                    cam.setGpioMode(1, tof.GpioMode.GENERAL_OUTPUT)
                    cam.setGpioMode(2, tof.GpioMode.GENERAL_OUTPUT)
                    gpio_enabled = True
                except RuntimeError as e:
                    messages_out.put(("log", f"could not configure GPIO ({e}). Running without outputs, only on screen."))

            step = "starting"
            cam.start()
            cam.switchConfig(mode)
            messages_out.put(("connected", None))
            try:
                messages_out.put(("modes", (list(cam.getMultipleNames()), mode)))
            except RuntimeError:
                messages_out.put(("modes", ([], mode)))
            failures = 0
            step = "reading frames"
            applied = None
            while True:
                while not commands_in.empty():
                    command, value = commands_in.get_nowait()
                    if command == "stop":
                        if gpio_enabled:
                            cam.setGpio(1, 1)
                            cam.setGpio(2, 1)
                        cam.stop()
                        return
                    if command == "outputs":
                        outputs = value
                    if command == "mode":
                        try:
                            cam.switchConfig(value)
                            mode = value
                            messages_out.put(("mode", mode))
                        except RuntimeError as e:
                            messages_out.put(("log", f"could not change to mode {value} ({e})."))
                # OUT1 goes low (0) while a wheel is outside the scale, OUT2 while the truck is completely inside
                if gpio_enabled and outputs != applied:
                    cam.setGpio(1, 0 if outputs == OUTSIDE else 1)
                    cam.setGpio(2, 0 if outputs == INSIDE else 1)
                    applied = outputs

                frames = None
                while cam.hasFrames():
                    frames = cam.getFrames()
                if frames is not None:
                    frames = frames_by_type(frames)
                    if all(int(t) in frames for t in types):
                        # Copied as arrays here: the camera reuses the memory of its frames
                        arrays = [np.asarray(frames[int(t)]).astype(np.float32) for t in frame_types]
                        if len(types) > len(frame_types):
                            color = color_to_image(np.asarray(frames[int(types[-1])]), arrays[2].shape)
                            if color is not None:
                                arrays.append(color)
                        try:
                            frames_out.put_nowait(arrays)
                        except queue.Full:
                            pass  # The main program is busy: this image is skipped, a newer one comes soon
                time.sleep(0.002)

        except RuntimeError as e:
            failures += 1
            if failures == 1:
                messages_out.put(("lost", f"{step}: {e}"))
            if 2 <= failures <= 3 or failures % 6 == 0:
                try:
                    found = ", ".join(f"{c.serial()} [{c.info()}]" for c in tof.discoverCameras()) or "none"
                except RuntimeError as d:
                    found = f"discovery failed ({d})"
                messages_out.put(("log", f"could not reconnect, failed {step} ({e}). Attempt {failures}. Sensors seen on the network: {found}"))
                if failures >= 6:
                    messages_out.put(("log", "if it never comes back, unplug the sensor's power for 10 s and plug it back in."))
        # Release the old connection completely before opening a new one to the same sensor
        if cam is not None:
            try:
                cam.stop()
            except RuntimeError:
                pass
        del cam
        gc.collect()

        # Wait before reconnecting, but quit right away if the main program asks
        waited = time.time()
        while time.time() - waited < RECONNECT_SECONDS:
            while not commands_in.empty():
                command, value = commands_in.get_nowait()
                if command == "stop":
                    return
                if command == "outputs":
                    outputs = value
                if command == "mode":
                    mode = value
            time.sleep(0.1)


class Sensor:
    ''' One 3D sensor: its own process, window, area corners, reference, floor plane and state '''

    def __init__(self, index, serial, rotation):
        self.index = index
        self.serial = serial
        self.window = f"Sensor {index + 1} ({serial})"
        self.rotation = rotation
        self.connected = False
        self.lost_time = time.time()   # Not connected yet (None = connected)
        self.has_intensity = True
        self.views = ["combined", "depth", "intensity"]   # "color" is added if the sensor sends it
        self.view = 0
        self.mode = CAMERA_MODE
        self.mode_names = []    # Names of the sensor's modes, sent by its process
        self.button = None      # Area (x0, y0, x1, y1) of the "Borrar marcas" button in the window
        self.frame = np.zeros((672, 804, 3), dtype=np.uint8)
        self.frames_in = mp.Queue(maxsize=2)
        self.messages = mp.Queue()
        self.commands = mp.Queue()
        self.process = mp.Process(target=sensor_process, args=(serial, self.frames_in, self.messages, self.commands), daemon=True)
        self.process.start()
        cv2.namedWindow(self.window)
        self.reset()

    def connection_lost(self, error):
        ''' The sensor stopped answering: keep its area and reference (it didn't move), its process reconnects it '''
        self.lost_time = time.time()
        self.result = None
        self.history.clear()
        log(f"{self.window}: CONNECTION LOST ({error}). Retrying every {RECONNECT_SECONDS} s; the other sensor keeps monitoring.")

    def reset(self):
        ''' Forget the area and the reference (new area, or the image was rotated) '''
        self.history = deque(maxlen=max(SMOOTHING_FRAMES, 1))
        self.corners = []
        self.reference = None
        self.reference_buffer = []
        self.capturing_reference = False
        self.calibrated_corners = None
        self.plane = None
        self.zone = None
        self.side = None
        self.result = None

    def clear_marks(self):
        ''' Erase the corners (button "Borrar marcas", key 'b' or right click). The reference of the empty
        area is kept: with 4 new corners the sensor measures again without pressing 'f' '''
        self.corners = []
        self.calibrated_corners = None
        self.plane = None
        self.zone = None
        self.side = None
        self.result = None

    def mode_name(self):
        return self.mode_names[self.mode] if self.mode < len(self.mode_names) else f"mode {self.mode}"

    def next_mode(self):
        ''' Ask the sensor's process for the next mode (key 'm') '''
        if self.lost_time is not None:
            return
        count = len(self.mode_names) or 7
        self.commands.put(("mode", (self.mode + 1) % count))

    def set_outputs(self, state):
        ''' Sent to the sensor's process: OUT1 = 0 while a wheel is outside, OUT2 = 0 while the truck is inside '''
        self.commands.put(("outputs", state))

    def read_frames(self):
        ''' Messages of the sensor's process and its latest frames (None = no new image yet), rotated for display '''
        while not self.messages.empty():
            kind, text = self.messages.get_nowait()
            if kind == "connected":
                self.lost_time = None
                log(f"{self.window}: {'reconnected' if self.connected else 'streaming'}.")
                self.connected = True
            elif kind == "lost":
                self.connection_lost(text)
            elif kind == "modes":
                self.mode_names, self.mode = text
            elif kind == "mode":
                self.mode = text
                # Each mode measures a bit differently: the reference of the empty area must be taken again
                self.history.clear()
                self.reference = None
                self.reference_buffer = []
                self.capturing_reference = False
                self.calibrated_corners = None
                self.plane = None
                self.zone = None
                self.result = None
                log(f"{self.window}: mode {self.mode} ({self.mode_name()}). The corners are kept; with the area empty press 'f' again."
                    f" To keep this mode, set CAMERA_MODE = {self.mode} at the top of the file.")
            else:
                log(f"{self.window}: {text}")
        arrays = None
        while not self.frames_in.empty():
            try:
                arrays = self.frames_in.get_nowait()
            except queue.Empty:
                break
        if arrays is None or self.lost_time is not None:
            return None
        return [np.ascontiguousarray(np.rot90(a, -(1 + self.rotation))) for a in arrays]

    def update(self):
        ''' Read the sensor, check the area and prepare the image to display '''
        arrays = self.read_frames()
        if self.lost_time is not None:
            self.frame = np.zeros_like(self.frame)
            text = "SIN CONEXION - reconectando..." if self.connected else "Conectando..."
            cv2.putText(self.frame, text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        if arrays is None:
            return  # No new image yet: keep showing the last one
        x, y, z = smooth_xyz(*arrays[:3], self.history)
        intensity = arrays[3] if self.has_intensity else None
        color = arrays[4] if len(arrays) > 4 else None
        if color is not None and "color" not in self.views:
            # First color image: start in the color view, the tape is easier to see there
            self.views.insert(0, "color")
            self.view = 0
            log(f"{self.window}: color view available (key 'v' changes the view).")

        if self.capturing_reference:
            self.reference_buffer.append([x, y, z])
            if len(self.reference_buffer) >= REFERENCE_FRAMES:
                self.reference = capture_reference(self.reference_buffer)
                self.capturing_reference = False
                self.calibrated_corners = None  # Forces the calibration below

        # Floor plane and area, computed again if the corners or the reference change
        if self.reference is not None and len(self.corners) == 4 and self.corners != self.calibrated_corners:
            self.calibrate(z.shape)

        if self.views[self.view] == "color" and color is not None:
            frame = color.copy()
        else:
            frame = render_view(self.views[self.view], z, intensity)

        if self.zone is not None:
            self.result, wheel_mask, outside_mask = check_wheels(x, y, z, self.reference, self.plane, self.side)

            # Wheels on the scale in green, the part past the edge in red
            overlay = frame.copy()
            overlay[wheel_mask] = (0, 200, 0)
            overlay[outside_mask] = (0, 0, 255)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Draw the area's corners and edges
        for c in self.corners:
            cv2.circle(frame, c, 4, (0, 255, 255), -1)
        if len(self.corners) >= 2:
            cv2.polylines(frame, [np.array(self.corners, dtype=np.int32)], len(self.corners) == 4, (0, 255, 255), 2)
        if self.side is not None:
            # Thick magenta: the edge this sensor measures (its side of the scale)
            i = self.side["index"]
            cv2.line(frame, self.corners[i], self.corners[(i + 1) % 4], (255, 0, 255), 5)

        self.frame = frame

    def calibrate(self, shape):
        ''' Floor plane and the area's corners on the floor, from the reference of the empty area '''
        zone_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(zone_mask, [np.array(self.corners, dtype=np.int32)], 1)
        self.plane = fit_floor_plane(*self.reference, zone_mask.astype(bool))
        self.zone = None
        if self.plane is None:
            log(f"{self.window}: the floor has no valid readings inside the area. Check the sensor and the corners.")
        else:
            zone = [pixel_to_floor(self.reference, self.plane, cx, cy) for cx, cy in self.corners]
            if any(c is None for c in zone):
                log(f"{self.window}: a corner has no valid reading. Click it again a bit inside the tape.")
            else:
                self.zone = zone
                self.side = scale_side(zone, self.plane)
                sides = [np.hypot(*np.subtract(zone[i], zone[(i + 1) % 4])) for i in range(4)]
                log(f"{self.window}: scale measured, sides " + ", ".join(f"{s:.0f}" for s in sides) + " mm.")
                i = self.side["index"]
                log(f"{self.window}: measures the edge between corners {i + 1} and {(i + 1) % 4 + 1} (magenta line),"
                    f" scale width {self.side['width']:.0f} mm. Monitoring.")
        self.calibrated_corners = list(self.corners)
        self.result = None

    def message(self):
        ''' Instruction or measure of this sensor, for its window '''
        if self.lost_time is not None:
            return "Sin conexion"
        if self.zone is not None:
            if self.result is None:
                return "No ve ruedas"
            return f"Margen de su lado: {self.result['margin']:.0f} mm" + (" | rueda pasada del extremo" if self.result["end_out"] else "")
        if self.capturing_reference:
            return "Guardando referencia..."
        if len(self.corners) < 4:
            return f"Marca las esquinas de la balanza ({len(self.corners)}/4)"
        return "Con la balanza vacia, presiona 'f'"

    def show(self, combined, width, active):
        ''' Display this sensor's image with its own measure and the combined result of both sensors '''
        frame = self.frame
        highlight_text(frame, f"S{self.index + 1}: {self.message()}", (10, 30), font_scale=0.6, text_color=(0, 0, 0), highlight_color=(255, 255, 255))
        colors = {OUTSIDE: ((0, 0, 255), (0, 255, 255)), INSIDE: ((0, 0, 0), (0, 255, 0)), DOUBTFUL: ((0, 0, 0), (0, 165, 255))}
        if combined in colors:
            highlight_text(frame, STATE_TEXT[combined], (10, 70), font_scale=0.9, text_color=colors[combined][0], highlight_color=colors[combined][1])
        if width is not None:
            highlight_text(frame, f"Ancho medido: {width:.0f} mm (camion {TRUCK_WIDTH_MM} mm)", (10, 105), font_scale=0.6,
                           text_color=(0, 0, 0), highlight_color=(255, 255, 255))
        # "Borrar marcas" button, top right (a click on it erases the corners)
        x1, y0 = frame.shape[1] - 10, 10
        x0, y1 = x1 - 170, y0 + 34
        cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 220), cv2.FILLED)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.putText(frame, "Borrar marcas", (x0 + 12, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.button = (x0, y0, x1, y1)

        # Sensor mode, at the bottom (key 'm' changes it)
        highlight_text(frame, f"Modo {self.mode}: {self.mode_name()}  (m = cambiar)", (10, frame.shape[0] - 15), font_scale=0.55,
                       text_color=(0, 0, 0), highlight_color=(255, 255, 255))
        if active:
            # Yellow border: the keys act on this sensor
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 255, 255), 4)
        cv2.imshow(self.window, frame)

    def stop(self):
        ''' Outputs back to 1 and the sensor stopped by its own process '''
        self.commands.put(("stop", None))
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()


active_sensor = 0  # Sensor the keys act on: the last window the mouse was over


def mouse_event(event, x, y, flags, sensor):
    '''
    Any mouse event selects this sensor. Left click: add a corner (4 in total), or erase them on the
    "Borrar marcas" button. Right click: erase the corners.
    '''
    global active_sensor
    active_sensor = sensor.index
    on_button = sensor.button is not None and sensor.button[0] <= x <= sensor.button[2] and sensor.button[1] <= y <= sensor.button[3]
    if (event == cv2.EVENT_LBUTTONDOWN and on_button) or event == cv2.EVENT_RBUTTONDOWN:
        sensor.clear_marks()
        log(f"{sensor.window}: corners erased. Click the 4 corners again.")
    elif event == cv2.EVENT_LBUTTONDOWN and len(sensor.corners) < 4:
        sensor.corners.append((x, y))


if __name__ == "__main__":
    serials = list(SERIALS)
    if not serials:
        serials = [input("Enter sensor 1 serial number: ").strip()]
        second = input("Enter sensor 2 serial number (empty = only one sensor): ").strip()
        if second:
            serials.append(second)

    # Each sensor runs in its own process, which connects it (and reconnects it after a cut)
    sensors = []
    for i, serial in enumerate(serials):
        rotation = IMAGE_ROTATIONS[i] if i < len(IMAGE_ROTATIONS) else 0
        log(f"Connecting sensor {i + 1} ({serial})...")
        sensors.append(Sensor(i, serial, rotation))
        cv2.setMouseCallback(sensors[-1].window, mouse_event, sensors[-1])

    combined, candidate, candidate_count = EMPTY, EMPTY, 0

    print("For EACH sensor window:")
    print("1. With the scale EMPTY, click its 4 corners (the tape) in order around it. The 'Borrar marcas' button (or b) erases them.")
    print("2. Press 'f' to save the empty scale as reference. The magenta line is the side that sensor measures.")
    print("Keys act on the window the mouse is over (yellow border):")
    print("f = save reference, n = reset, v = change view, m = change sensor mode, b = erase corners, r = rotate image 90°, q = quit.")

    while True:
        for s in sensors:
            s.update()

        # Result of the scale with all the sensors; a change is confirmed after several consecutive frames
        measuring = [s for s in sensors if s.zone is not None and s.lost_time is None]
        current, width, out_sides = combine_results(measuring)
        if current == candidate:
            candidate_count += 1
        else:
            candidate, candidate_count = current, 1
        if candidate != combined and candidate_count >= TEMPORAL_THRESHOLD:
            combined = candidate
            margins = ", ".join(f"S{s.index + 1} {s.result['margin']:.0f} mm" for s in measuring if s.result is not None)
            detail = f" (margins: {margins}" + (f", width {width:.0f} mm" if width is not None else "") + ")" if margins else ""
            sides = f" Side of sensor {', '.join(map(str, out_sides))}." if out_sides else ""
            log(f"SCALE: {STATE_TEXT[combined]}{detail}{sides}")
            for s in sensors:
                s.set_outputs(combined)

        for s in sensors:
            s.show(combined, width, s.index == active_sensor and len(sensors) > 1)

        # Read the keyboard only once per frame so key presses aren't lost
        key = cv2.waitKey(1) & 0xFF
        sensor = sensors[active_sensor]

        if key == ord('q'):
            break

        if key == ord('v'):
            sensor.view = (sensor.view + 1) % len(sensor.views)

        if key == ord('m'):
            sensor.next_mode()

        if key == ord('b'):
            sensor.clear_marks()
            log(f"{sensor.window}: corners erased. Click the 4 corners again.")

        # Rotating the image moves every pixel, so the corners and the reference must be taken again
        if key == ord('r'):
            sensor.rotation = (sensor.rotation + 1) % 4
            log(f"{sensor.window}: image rotated {sensor.rotation * 90}°. To keep it, set IMAGE_ROTATIONS[{sensor.index}] = {sensor.rotation} at the top of the file.")

        if key in (ord('n'), ord('r')):
            sensor.reset()
            log(f"{sensor.window}: reset. Click the 4 corners of the empty area.")

        if key == ord('f'):
            if len(sensor.corners) < 4:
                log(f"{sensor.window}: first click the 4 corners of the area.")
            else:
                sensor.reference_buffer = []
                sensor.capturing_reference = True
                log(f"{sensor.window}: capturing reference, the area must be empty...")

    for s in sensors:
        s.stop()
    cv2.destroyAllWindows()
