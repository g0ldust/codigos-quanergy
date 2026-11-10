import warnings
import cv2
import numpy as np
import chronoptics.tof as tof
from collections import deque
from datetime import datetime

DIFF_THRESHOLD_MM = 50       # A pixel counts as changed if its depth differs this much (mm) from the reference
TEMPORAL_THRESHOLD = 5       # Number of consecutive frames to confirm a change of state
REFERENCE_FRAMES = 10        # Frames averaged to capture the reference of the object in place
CAMERA_MODE = 5              # Medium Range Static Scene
IMAGE_ROTATION = 0           # Extra 90° clockwise turns of the image (2 = sensor mounted upside down). Key 'r' changes it
SMOOTHING_FRAMES = 5         # Frames averaged per pixel to reduce flicker (1 = off)
MEDIAN_FILTER_SIZE = 5       # Median filter to remove isolated noisy pixels (3 or 5, 0 = off)
SHOW_COLOR_CAMERA = True     # Show the sensor's color camera in its own window (only models with camera)
COLOR_ROTATION = 0           # Extra 90° clockwise turns of the color camera image. Key 'c' changes it


def list_camera_modes(cam):
    ''' Print the sensor's modes, to try another CAMERA_MODE if the image is noisy '''
    try:
        names = cam.getMultipleNames()
        descriptions = cam.getMultipleDescriptions()
    except RuntimeError:
        return
    print("Camera modes (CAMERA_MODE):")
    for i, (name, description) in enumerate(zip(names, descriptions)):
        name = name.decode("ascii", "ignore") if isinstance(name, bytes) else name
        description = description.decode("ascii", "ignore") if isinstance(description, bytes) else description
        print(f"  {'*' if i == CAMERA_MODE else ' '} {i} {name}: {description}")


def smooth_depth(depth, history):
    '''
    Reduce noise: per-pixel average of the last frames (only valid readings; a pixel valid in less than
    half of them is set invalid, which removes flicker), then a median filter for isolated noisy pixels.
    '''
    history.append(depth)
    if len(history) > 1:
        stack = np.stack(history)
        count = np.count_nonzero(stack > 0, axis=0)
        mean = stack.sum(axis=0) / np.maximum(count, 1)  # Invalid readings are 0, so they don't add to the sum
        depth = np.where(count >= (len(history) + 1) // 2, mean, 0).astype(np.float32)
    if MEDIAN_FILTER_SIZE in (3, 5):
        depth = cv2.medianBlur(depth, MEDIAN_FILTER_SIZE)
    return depth


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
    - "combined": depth colors shaded with the intensity, shows the distance and also the details (edges, textures).
    - "depth": only the depth colors.
    - "intensity": the infrared image in gray, like a normal camera.
    '''
    if view == "intensity" and intensity is not None:
        return cv2.cvtColor(intensity_to_gray(intensity), cv2.COLOR_GRAY2BGR)
    colored = depth_to_color(depth)
    if view == "combined" and intensity is not None:
        shade = 0.25 + 0.75 * intensity_to_gray(intensity).astype(np.float32) / 255
        colored = (colored * shade[..., None]).astype(np.uint8)
    return colored


def get_color_image(frame, rotation=0):
    ''' Color camera image (BGR) ready for OpenCV: it comes turned 180° (upside down and mirrored) '''
    image = np.rot90(np.asarray(frame), 2).astype(np.uint8)
    return np.ascontiguousarray(np.rot90(image, -rotation))


def frames_by_type(frames):
    ''' Frames indexed by their type: the camera doesn't always return them in the order they were requested '''
    return {int(f.frameType()): f for f in frames}


def get_frame_array(frame, rotation=0):
    '''
    Frame as a contiguous array rotated to the sensor's display orientation, plus `rotation` extra
    90° clockwise turns. The frame is read in its own data type and converted afterwards: the SDK can't
    convert it with NumPy 2.
    '''
    return np.ascontiguousarray(np.rot90(np.asarray(frame).astype(np.float32), -(1 + rotation)))


def capture_reference(frames_list, rect):
    '''
    Reference of the ROI with the object in place: per-pixel median of several frames,
    ignoring invalid (zero) readings.
    '''
    x1, y1, x2, y2 = rect
    stack = np.stack([f[y1:y2, x1:x2] for f in frames_list]).astype(np.float32)
    stack[stack <= 0] = np.nan
    # Pixels with no valid reading in any frame give NaN (silenced warning) and are set to 0 (invalid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reference = np.nanmedian(stack, axis=0)
    return np.nan_to_num(reference, nan=0.0)


def changed_ratio(depth_frame, reference, rect, diff_threshold=DIFF_THRESHOLD_MM):
    '''
    Fraction (0-1) of the ROI's valid reference pixels whose depth changed.
    A pixel that had a valid reading and now has none also counts as changed.
    '''
    x1, y1, x2, y2 = rect
    current = depth_frame[y1:y2, x1:x2]
    valid_reference = reference > 0
    if not np.any(valid_reference):
        return 0.0
    changed = valid_reference & ((current <= 0) | (np.abs(current - reference) > diff_threshold))
    return np.sum(changed) / np.sum(valid_reference)


# Function to highlight text with background
def highlight_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, text_color=(0, 0, 255), highlight_color=(0, 255, 255)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position

    # Draw a filled rectangle behind the text (highlight color)
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), highlight_color, thickness=cv2.FILLED)

    # Draw the text on top of the rectangle (text color)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)


# Global variables for ROI selection with the mouse
draw = False
roi_selected = False
rect = (0, 0, 0, 0)
start_point = (0, 0)


def click_event(event, x, y, flags, param):
    global start_point, draw, rect, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        draw = True
        roi_selected = False
    elif event == cv2.EVENT_LBUTTONUP and draw:
        draw = False
        x1, x2 = sorted((start_point[0], x))
        y1, y2 = sorted((start_point[1], y))
        if x2 - x1 > 5 and y2 - y1 > 5:  # Ignore simple clicks
            rect = (max(x1, 0), max(y1, 0), x2, y2)
            roi_selected = True


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


gpio_enabled = True  # False if the camera doesn't accept the GPIO configuration


def set_output(cam, value):
    ''' Set output pin 1 (0 = object missing, 1 = idle), only if GPIO could be configured '''
    if gpio_enabled:
        cam.setGpio(1, value)


if __name__ == "__main__":
    # Initialize 3D camera
    serial = input("Enter sensor's serial number: ")
    cam = tof.TuiCamera(serial=serial)

    # Z for detecting; intensity (if available) only for a more detailed image; color camera (if the model has it)
    has_intensity = hasattr(tof.FrameType, "INTENSITY")
    frame_types = [tof.FrameType.Z, tof.FrameType.INTENSITY] if has_intensity else [tof.FrameType.Z]
    has_color = False
    if SHOW_COLOR_CAMERA:
        try:
            has_color = tof.FrameType.BGR in cam.possibleFrameTypes()
        except RuntimeError:
            pass
        if not has_color:
            log("This sensor has no color camera: showing only the 3D image.")
    try:
        cam.setOutputFrameTypes(frame_types + [tof.FrameType.BGR] if has_color else frame_types)
    except RuntimeError as e:
        has_color = False
        cam.setOutputFrameTypes(frame_types)
        log(f"Warning: could not enable the color camera ({e}). Showing only the 3D image.")

    # Continuous is the camera's default trigger mode: if the camera doesn't accept the setting, keep going
    try:
        cam.setTriggerMode(tof.TriggerMode.CONTINUOUS)
    except RuntimeError as e:
        log(f"Warning: could not set the trigger mode ({e}), using the camera's default.")

    # Configure GPIO: OUT1 goes low (0) while the object is missing, like the Intrusion Detection applet
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
    list_camera_modes(cam)

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_event)
    if has_color:
        cv2.namedWindow("Camara", cv2.WINDOW_NORMAL)  # Resizable: the color image is larger than the screen
        cv2.resizeWindow("Camara", 960, 540)
    color_rotation = COLOR_ROTATION

    change_threshold = 0.30      # Fraction of the ROI that must change to consider the object removed
    reference = None             # Depth of the ROI with the object in place
    reference_rect = None        # ROI the reference was captured with
    reference_buffer = []        # Frames collected while capturing the reference
    capturing_reference = False
    object_missing = False
    consecutive_changes = 0
    consecutive_ok = 0
    rotation = IMAGE_ROTATION
    depth_history = deque(maxlen=max(SMOOTHING_FRAMES, 1))  # Last frames, for the noise filter
    views = ["combined", "depth", "intensity"] if has_intensity else ["depth"]
    view = 0

    print("1. Drag with the mouse a rectangle tightly around the object.")
    print("2. With the object in place, press 'f' to save its position.")
    print("Keys: f = save reference, n = new area, + / - = sensitivity, r = rotate image 90°, v = change view, "
          "c = rotate color camera 90°, q = quit.")

    while cam.isStreaming():
        frames = frames_by_type(cam.getFrames())

        # The color camera can run at a different rate: show it whenever a color frame arrives
        if has_color and tof.FrameType.BGR in frames:
            cv2.imshow("Camara", get_color_image(frames[tof.FrameType.BGR], color_rotation))
        if tof.FrameType.Z not in frames:
            cv2.waitKey(1)
            continue

        depth_frame = smooth_depth(get_frame_array(frames[tof.FrameType.Z], rotation), depth_history)
        intensity = get_frame_array(frames[tof.FrameType.INTENSITY], rotation) if has_intensity else None

        # Read the keyboard only once per frame so key presses aren't lost
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c') and has_color:
            color_rotation = (color_rotation + 1) % 4
            log(f"Color camera rotated {color_rotation * 90}°. To keep it, set COLOR_ROTATION = {color_rotation} at the top of the file.")

        if key == ord('v'):
            view = (view + 1) % len(views)
            log(f"View: {views[view]}")

        # depth converted to a visual format for display
        frame = render_view(views[view], depth_frame, intensity)

        # Rotating the image moves every pixel, so the area and the reference must be taken again
        if key == ord('r'):
            rotation = (rotation + 1) % 4
            depth_history.clear()  # Old frames have the previous orientation
            log(f"Image rotated {rotation * 90}°. To keep it, set IMAGE_ROTATION = {rotation} at the top of the file.")

        if key in (ord('n'), ord('r')):
            roi_selected = False
            reference = None
            reference_rect = None
            capturing_reference = False
            if object_missing:
                set_output(cam, 1)
            object_missing = False
            log("Reset. Select a new area.")

        if key in (ord('+'), ord('=')):
            change_threshold = min(change_threshold + 0.05, 0.95)
            log(f"Sensitivity threshold: {change_threshold*100:.0f}% of the area")
        elif key in (ord('-'), ord('_')):
            change_threshold = max(change_threshold - 0.05, 0.05)
            log(f"Sensitivity threshold: {change_threshold*100:.0f}% of the area")

        if not roi_selected:
            highlight_text(frame, "Selecciona el objeto con el mouse", (10, 30), font_scale=0.7, text_color=(0, 0, 0), highlight_color=(255, 255, 255))
        else:
            x1, y1, x2, y2 = rect
            x2, y2 = min(x2, depth_frame.shape[1]), min(y2, depth_frame.shape[0])
            rect = (x1, y1, x2, y2)

            # A new area was selected with the mouse: the old reference no longer applies
            if reference is not None and rect != reference_rect:
                reference = None
                reference_rect = None
                if object_missing:
                    set_output(cam, 1)
                object_missing = False
                log("New area selected. Press 'f' to save the reference.")

            if key == ord('f'):
                reference_buffer = []
                capturing_reference = True
                log("Capturing reference, keep the object still...")

            if capturing_reference:
                reference_buffer.append(depth_frame)
                highlight_text(frame, "Guardando referencia...", (10, 30), font_scale=0.7, text_color=(0, 0, 0), highlight_color=(255, 255, 255))
                if len(reference_buffer) >= REFERENCE_FRAMES:
                    reference = capture_reference(reference_buffer, rect)
                    reference_rect = rect
                    capturing_reference = False
                    object_missing = False
                    consecutive_changes = consecutive_ok = 0
                    set_output(cam, 1)
                    valid = np.mean(reference > 0) * 100
                    log(f"Reference saved ({valid:.0f}% of the area has a valid reading). Monitoring object.")
                    if valid < 50:
                        log("Warning: less than half of the area has a valid reading, detection may be unreliable.")

            if reference is None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                if not capturing_reference:
                    highlight_text(frame, "Presiona 'f' con el objeto en su lugar", (10, 30), font_scale=0.7, text_color=(0, 0, 0), highlight_color=(255, 255, 255))
            else:
                ratio = changed_ratio(depth_frame, reference, rect)

                if ratio > change_threshold:
                    consecutive_changes += 1
                    consecutive_ok = 0
                else:
                    consecutive_ok += 1
                    consecutive_changes = 0

                if not object_missing and consecutive_changes >= TEMPORAL_THRESHOLD:
                    object_missing = True
                    set_output(cam, 0)  # If the object is removed, set output pin 1 to low
                    log("OBJECT REMOVED")
                elif object_missing and consecutive_ok >= TEMPORAL_THRESHOLD:
                    object_missing = False
                    set_output(cam, 1)
                    log("Object back in place")

                if object_missing:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    highlight_text(frame, "OBJETO RETIRADO!", (10, 30), font_scale=1, text_color=(0, 0, 255), highlight_color=(0, 255, 255))
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    highlight_text(frame, "Objeto en su lugar", (10, 30), font_scale=0.8, text_color=(0, 0, 0), highlight_color=(0, 255, 0))

                cv2.putText(frame, f"Cambio: {ratio*100:.0f}%  (umbral {change_threshold*100:.0f}%)", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)

    set_output(cam, 1)
    cam.stop()
    cv2.destroyAllWindows()
    cam = None
