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

# Scale model: the 60 x 60 x 35 cm box is the truck (2.2 m wide -> scale 1 : 3.67), the scale 82 cm wide
# (real truck: TOLERANCE_MM = 400, CONSISTENCY_MM = 200)
TOLERANCE_MM = 110           # How much the truck can move from the taught position before a wheel leaves the scale:
                             # (scale width - truck width) / 2 = (818 - 600) / 2 in the model, (3000 - 2200) / 2 in real
CONSISTENCY_MM = 60          # With two sensors facing each other, one distance goes down as much as the other goes up.
                             # If the two changes don't match by more than this, something else is in the way: doubtful
CHANGE_THRESHOLD_MM = 50     # A point counts as "truck" if its depth changed this much (mm) from the empty reference
MIN_OBJECT_PIXELS = 150      # Points needed to consider that a sensor sees the truck
NEAR_PERCENTILE = 5          # The truck's distance is the face closest to the sensor (this percentile of its points)
FLOOR_CORRECTION = True      # Measure the distance along the floor (corrects sensors that look down at an angle)
TEMPORAL_THRESHOLD = 5       # Number of consecutive frames to confirm a change of state
REFERENCE_FRAMES = 10        # Frames combined for the empty reference and for the taught position
CAMERA_MODE = 5              # Medium Range Static Scene
SMOOTHING_FRAMES = 5         # Frames averaged per pixel to reduce flicker (1 = off)
MEDIAN_FILTER_SIZE = 5       # Median filter to remove isolated noisy pixels (3 or 5, 0 = off)
USE_GPIO = False             # True = use the outputs OUT1/OUT2. Off: configuring them broke the network connection of the sensors
RECONNECT_SECONDS = 5        # If a sensor loses the connection, try to reconnect it every this many seconds

# States of the scale
EMPTY, INSIDE, OUTSIDE, DOUBTFUL = "empty", "inside", "outside", "doubtful"
STATE_TEXT = {EMPTY: "Balanza vacia", INSIDE: "DENTRO DE LA BALANZA", OUTSIDE: "FUERA DE LA BALANZA!",
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


def view_direction(reference, roi):
    '''
    Direction in which this sensor measures the distance to the truck. It is the sensor's view axis (Z); if
    FLOOR_CORRECTION is on and the floor inside the rectangle can be fitted, it is the view axis laid flat
    on the floor, so a sensor that looks down at an angle measures how much the truck moves on the floor,
    not along its tilted axis. Returns (direction, tilt in degrees or None if not corrected).
    '''
    axis = np.array([0.0, 0.0, 1.0])
    if not FLOOR_CORRECTION:
        return axis, None
    plane = fit_floor_plane(*reference, roi)
    if plane is None:
        return axis, None
    normal = plane[1]
    tilt = float(np.degrees(np.arcsin(min(abs(axis @ normal), 1.0))))
    if tilt > 70:
        # Looking almost straight at the fitted plane: it is probably a wall, not the floor
        return axis, None
    flat = axis - (axis @ normal) * normal
    return flat / np.linalg.norm(flat), tilt


def measure_distance(x, y, z, reference, roi, direction):
    '''
    Distance (mm) from the sensor to the face of the truck closest to it, along `direction`.
    The truck is what changed from the empty reference inside the rectangle. Returns (distance or None if
    the sensor doesn't see the truck, mask of the truck's points, mask of the measured face).
    '''
    z_ref = reference[2]
    obj = roi & (z > 0) & (z_ref > 0) & (np.abs(z - z_ref) > CHANGE_THRESHOLD_MM)
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    face = np.zeros(z.shape, dtype=bool)
    if obj.sum() < MIN_OBJECT_PIXELS:
        return None, obj, face
    along = x * direction[0] + y * direction[1] + z * direction[2]
    distance = float(np.percentile(along[obj], NEAR_PERCENTILE))
    face = obj & (along < distance + 20)
    return distance, obj, face


def combine_positions(sensors):
    '''
    Result of the scale from the sensors that are measuring (empty reference and taught position done).
    change = current distance - taught distance (+ = the truck moved away from that sensor).
    - No sensor sees the truck: empty.
    - Two sensors facing each other: the truck moved (change 1 - change 2) / 2 towards sensor 2. If the two
      changes don't mirror each other (change 1 + change 2 far from 0), something else is in the way: doubtful.
    - Moved more than TOLERANCE_MM: outside. Otherwise: inside.
    Returns (state, shift in mm towards the last sensor or None, text with the detail).
    '''
    seen = [s for s in sensors if s.change is not None]
    if not seen:
        return EMPTY, None, ""
    if len(seen) == 2:
        c1, c2 = seen[0].change, seen[1].change
        shift = (c1 - c2) / 2
        if abs(c1 + c2) > CONSISTENCY_MM:
            return DOUBTFUL, shift, f"los cambios no coinciden ({c1 / 10:+.1f} y {c2 / 10:+.1f} cm)"
        towards = seen[1] if shift > 0 else seen[0]
    else:
        # Only one sensor: moving away from it means moving towards the other side
        shift = seen[0].change
        towards = None
    text = f"corrida {abs(shift) / 10:.1f} cm" + (f" hacia S{towards.index + 1}" if towards is not None and abs(shift) >= 5 else "")
    text += f" (tolerancia {TOLERANCE_MM / 10:.0f} cm)"
    return (OUTSIDE if abs(shift) > TOLERANCE_MM else INSIDE), shift, text


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

            # GPIO: OUT1 = truck outside the scale, OUT2 = truck inside (both active low, like the other applets)
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
                # OUT1 goes low (0) while the truck is outside the scale, OUT2 while the truck is completely inside
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
    ''' One 3D sensor: its own process, window, rectangle, empty reference, taught distance and measure '''

    def __init__(self, index, serial, rotation):
        self.index = index
        self.serial = serial
        self.window = f"Sensor {index + 1} ({serial})"
        self.rotation = rotation
        self.connected = False
        self.lost_time = time.time()   # Not connected yet (None = connected)
        self.views = ["combined", "depth", "intensity"]   # "color" is added if the sensor sends it
        self.view = 0
        self.mode = CAMERA_MODE
        self.mode_names = []    # Names of the sensor's modes, sent by its process
        self.button = None      # Area (x0, y0, x1, y1) of the "Borrar marcas" button in the window
        self.drag = None        # Rectangle being drawn with the mouse: (start, end)
        self.frame = np.zeros((672, 804, 3), dtype=np.uint8)
        self.frames_in = mp.Queue(maxsize=2)
        self.messages = mp.Queue()
        self.commands = mp.Queue()
        self.process = mp.Process(target=sensor_process, args=(serial, self.frames_in, self.messages, self.commands), daemon=True)
        self.process.start()
        cv2.namedWindow(self.window)
        self.reset()

    def connection_lost(self, error):
        ''' The sensor stopped answering: keep its rectangle, reference and taught distance; its process reconnects it '''
        self.lost_time = time.time()
        self.distance = self.change = None
        self.history.clear()
        log(f"{self.window}: CONNECTION LOST ({error}). Retrying every {RECONNECT_SECONDS} s; the other sensor keeps monitoring.")

    def reset(self):
        ''' Forget everything: rectangle, empty reference and taught position (key 'n', or the image was rotated) '''
        self.history = deque(maxlen=max(SMOOTHING_FRAMES, 1))
        self.rect = None
        self.forget_reference()

    def forget_reference(self):
        ''' Forget the empty reference and the taught position (the sensor's mode changed) '''
        self.reference = None
        self.reference_buffer = []
        self.capturing_reference = False
        self.direction, self.tilt = None, None
        self.forget_teaching()

    def forget_teaching(self):
        ''' Forget the taught position (the rectangle changed) '''
        self.taught = None
        self.teach_buffer = []
        self.teaching = False
        self.distance = self.change = None

    def clear_marks(self):
        ''' Erase the rectangle (button "Borrar marcas", key 'b' or right click). The empty reference is kept '''
        self.rect = None
        self.drag = None
        self.forget_teaching()

    def set_rect(self, start, end):
        ''' Rectangle where the truck can be, drawn with the mouse '''
        (x0, y0), (x1, y1) = start, end
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return
        self.rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.forget_teaching()
        self.direction = None   # Computed again with the new rectangle
        log(f"{self.window}: rectangle set. " + ("Put the box in its place and press 't'." if self.reference is not None
                                                 else "With the scale EMPTY press 'f'."))

    def roi_mask(self, shape):
        mask = np.zeros(shape, dtype=bool)
        x0, y0, x1, y1 = self.rect
        mask[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
        return mask

    def mode_name(self):
        return self.mode_names[self.mode] if self.mode < len(self.mode_names) else f"mode {self.mode}"

    def next_mode(self):
        ''' Ask the sensor's process for the next mode (key 'm') '''
        if self.lost_time is not None:
            return
        count = len(self.mode_names) or 7
        self.commands.put(("mode", (self.mode + 1) % count))

    def set_outputs(self, state):
        ''' Sent to the sensor's process: OUT1 = 0 while the truck is outside, OUT2 = 0 while it is inside '''
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
                # Each mode measures a bit differently: the empty reference and the position must be taken again
                self.history.clear()
                self.forget_reference()
                log(f"{self.window}: mode {self.mode} ({self.mode_name()}). The rectangle is kept; press 'f' (scale empty)"
                    f" and 't' (box in place) again. To keep this mode, set CAMERA_MODE = {self.mode} at the top of the file.")
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
        ''' Read the sensor, measure the distance to the truck and prepare the image to display '''
        arrays = self.read_frames()
        if self.lost_time is not None:
            self.frame = np.zeros_like(self.frame)
            text = "SIN CONEXION - reconectando..." if self.connected else "Conectando..."
            cv2.putText(self.frame, text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        if arrays is None:
            return  # No new image yet: keep showing the last one
        x, y, z = smooth_xyz(*arrays[:3], self.history)
        intensity = arrays[3]
        color = arrays[4] if len(arrays) > 4 else None
        if color is not None and "color" not in self.views:
            self.views.insert(0, "color")
            log(f"{self.window}: color view available (key 'v' changes the view).")

        if self.capturing_reference:
            self.reference_buffer.append([x, y, z])
            if len(self.reference_buffer) >= REFERENCE_FRAMES:
                self.reference = capture_reference(self.reference_buffer)
                self.capturing_reference = False
                self.direction = None
                log(f"{self.window}: empty reference saved. Put the box in its place and press 't'.")

        # Direction of the measure, computed again if the reference or the rectangle change
        if self.reference is not None and self.rect is not None and self.direction is None:
            self.direction, self.tilt = view_direction(self.reference, self.roi_mask(z.shape))
            log(f"{self.window}: " + (f"looks {self.tilt:.0f}° down at the floor, the distance is measured along the floor."
                                      if self.tilt is not None else "the distance is measured along the sensor's view axis (Z)."))

        if self.views[self.view] == "color" and color is not None:
            frame = color.copy()
        else:
            frame = render_view(self.views[self.view], z, intensity)

        self.distance = self.change = None
        if self.direction is not None:
            self.distance, obj, face = measure_distance(x, y, z, self.reference, self.roi_mask(z.shape), self.direction)

            # The truck in green, the measured face (closest to the sensor) in magenta
            overlay = frame.copy()
            overlay[obj] = (0, 200, 0)
            overlay[face] = (255, 0, 255)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            if self.teaching:
                self.teach_buffer.append(self.distance)
                if len(self.teach_buffer) >= REFERENCE_FRAMES:
                    seen = [d for d in self.teach_buffer if d is not None]
                    self.teaching = False
                    if len(seen) < len(self.teach_buffer) // 2:
                        log(f"{self.window}: it doesn't see the box inside the rectangle. Check the rectangle and press 't' again.")
                    else:
                        self.taught = float(np.median(seen))
                        log(f"{self.window}: position taught, distance to the box {self.taught / 10:.1f} cm. Monitoring.")
            if self.taught is not None and self.distance is not None:
                self.change = self.distance - self.taught

        # Rectangle where the truck can be (and the one being drawn)
        if self.rect is not None:
            cv2.rectangle(frame, self.rect[:2], self.rect[2:], (0, 255, 255), 2)
        if self.drag is not None:
            cv2.rectangle(frame, self.drag[0], self.drag[1], (0, 255, 255), 1)

        self.frame = frame

    def message(self):
        ''' Instruction or measure of this sensor, for its window '''
        if self.lost_time is not None:
            return "Sin conexion"
        if self.rect is None:
            return "Arrastra un rectangulo donde puede estar la caja"
        if self.capturing_reference:
            return "Guardando balanza vacia..."
        if self.reference is None:
            return "Con la balanza VACIA, presiona 'f'"
        if self.teaching:
            return "Guardando posicion..."
        if self.taught is None:
            seen = f" (la ve a {self.distance / 10:.1f} cm)" if self.distance is not None else " (no ve la caja)"
            return "Pon la caja bien ubicada y presiona 't'" + seen
        if self.distance is None:
            return f"No ve la caja (posicion: Z = {self.taught / 10:.1f} cm)"
        return f"Z = {self.distance / 10:.1f} cm  (posicion {self.taught / 10:.1f} cm, cambio {self.change / 10:+.1f} cm)"

    def show(self, combined, detail, active):
        ''' Display this sensor's image with its own measure and the combined result of both sensors '''
        frame = self.frame
        highlight_text(frame, f"S{self.index + 1}: {self.message()}", (10, 30), font_scale=0.6, text_color=(0, 0, 0), highlight_color=(255, 255, 255))
        colors = {OUTSIDE: ((0, 0, 255), (0, 255, 255)), INSIDE: ((0, 0, 0), (0, 255, 0)), DOUBTFUL: ((0, 0, 0), (0, 165, 255))}
        if combined in colors:
            highlight_text(frame, STATE_TEXT[combined], (10, 70), font_scale=0.9, text_color=colors[combined][0], highlight_color=colors[combined][1])
        if detail:
            highlight_text(frame, detail, (10, 105), font_scale=0.6, text_color=(0, 0, 0), highlight_color=(255, 255, 255))

        # "Borrar marcas" button, top right (a click on it erases the rectangle)
        x1, y0 = frame.shape[1] - 10, 45
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
    Any mouse event selects this sensor. Drag with the left button: rectangle where the truck can be.
    Click on "Borrar marcas" or right click: erase the rectangle.
    '''
    global active_sensor
    active_sensor = sensor.index
    on_button = sensor.button is not None and sensor.button[0] <= x <= sensor.button[2] and sensor.button[1] <= y <= sensor.button[3]
    if (event == cv2.EVENT_LBUTTONDOWN and on_button) or event == cv2.EVENT_RBUTTONDOWN:
        sensor.clear_marks()
        log(f"{sensor.window}: rectangle erased. Drag a new one where the box can be.")
    elif event == cv2.EVENT_LBUTTONDOWN:
        sensor.drag = ((x, y), (x, y))
    elif event == cv2.EVENT_MOUSEMOVE and sensor.drag is not None and flags & cv2.EVENT_FLAG_LBUTTON:
        sensor.drag = (sensor.drag[0], (x, y))
    elif event == cv2.EVENT_LBUTTONUP and sensor.drag is not None:
        sensor.set_rect(sensor.drag[0], (x, y))
        sensor.drag = None


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

    print("1. In EACH sensor window, drag a rectangle around where the box can be (bigger than the box).")
    print("   The 'Borrar marcas' button (or b, or right click) erases it.")
    print("2. With the scale EMPTY press 'f' (saves the empty scale on all the sensors).")
    print("3. Put the box in its right place and press 't' (all the sensors save their distance to the box).")
    print("Then each sensor shows its Z and how much it changed; if the box moves more than the tolerance: outside.")
    print("Keys: f = empty scale, t = teach position, v = change view, m = change sensor mode, b = erase rectangle,")
    print("      n = reset sensor, r = rotate image 90°, q = quit. v, m, b, n, r act on the window with the yellow border.")

    while True:
        for s in sensors:
            s.update()

        # Result of the scale with all the sensors; a change is confirmed after several consecutive frames
        measuring = [s for s in sensors if s.taught is not None and s.lost_time is None]
        current, shift, detail = combine_positions(measuring)
        if current == candidate:
            candidate_count += 1
        else:
            candidate, candidate_count = current, 1
        if candidate != combined and candidate_count >= TEMPORAL_THRESHOLD:
            combined = candidate
            distances = ", ".join(f"S{s.index + 1} Z = {s.distance / 10:.1f} cm ({s.change / 10:+.1f})" for s in measuring if s.change is not None)
            log(f"SCALE: {STATE_TEXT[combined]}" + (f" - {detail}" if detail else "") + (f" [{distances}]" if distances else ""))
            for s in sensors:
                s.set_outputs(combined)

        for s in sensors:
            s.show(combined, detail if measuring else "", s.index == active_sensor and len(sensors) > 1)

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
            log(f"{sensor.window}: rectangle erased. Drag a new one where the box can be.")

        # 'f' and 't' act on all the sensors at once: they must see the same scene (empty / box in place)
        if key == ord('f'):
            for s in sensors:
                if s.rect is None:
                    log(f"{s.window}: first drag the rectangle where the box can be.")
                elif s.lost_time is None:
                    s.forget_reference()
                    s.capturing_reference = True
            log("Saving the empty scale, it must be empty...")

        if key == ord('t'):
            for s in sensors:
                if s.reference is None or s.direction is None:
                    log(f"{s.window}: first the rectangle and the empty scale ('f').")
                elif s.lost_time is None:
                    s.forget_teaching()
                    s.teaching = True
            log("Saving the position of the box, don't move it...")

        # Rotating the image moves every pixel, so everything must be taken again
        if key == ord('r'):
            sensor.rotation = (sensor.rotation + 1) % 4
            log(f"{sensor.window}: image rotated {sensor.rotation * 90}°. To keep it, set IMAGE_ROTATIONS[{sensor.index}] = {sensor.rotation} at the top of the file.")

        if key in (ord('n'), ord('r')):
            sensor.reset()
            log(f"{sensor.window}: reset. Drag the rectangle where the box can be.")

    for s in sensors:
        s.stop()
    cv2.destroyAllWindows()
