import chronoptics.tof as tof
import numpy as np
import copy
import cv2
from box_measurements import calculate_box_dimensions
from fill_percentage import calculate_fill_percentage
from box_angle import calculate_box_angle


def get_xyz_arrays(frame):
    ''' Convert X, Y, Z frames to contiguous arrays rotated to the sensor's display orientation '''
    x_frame_array = np.ascontiguousarray(np.rot90(np.asarray(frame[0]), -1))
    y_frame_array = np.ascontiguousarray(np.rot90(np.asarray(frame[1]), -1))
    z_frame_array = np.ascontiguousarray(np.rot90(np.asarray(frame[2]), -1))
    return x_frame_array, y_frame_array, z_frame_array


def get_center_depth(z_frame_array, size=5):
    ''' Depth at the center of the frame (box's base), as the median of a small window to avoid zero/noisy pixels '''
    row_index = z_frame_array.shape[0] // 2
    col_index = z_frame_array.shape[1] // 2
    window = z_frame_array[row_index-size:row_index+size, col_index-size:col_index+size]
    window = window[window > 0]
    return float(np.median(window)) if window.size > 0 else 0.0


def get_search_mask(shape, search_rect=None, center_fraction=0.6):
    '''
    Area where the box is searched for: the rectangle selected with the mouse, or by default a
    circle in the center of the image, so the sensor's mount or objects at the borders are ignored.
    '''
    height, width = shape
    search_mask = np.zeros((height, width), dtype=np.uint8)
    if search_rect is not None:
        x1, y1, x2, y2 = search_rect
        cv2.rectangle(search_mask, (x1, y1), (x2, y2), 255, -1)
    else:
        center = (width // 2, height // 2)  # Center of the image
        radius = int(min(center[0], center[1]) * center_fraction)
        cv2.circle(search_mask, center, radius, 255, -1)  # Draw a filled circle
    return search_mask


def detect_box(z_frame_array, search_rect=None):
    '''
    Detect the box's top edges in the depth frame, only inside the search area.
    Returns (approx, mask): approx are the four vertices of the box (None if no box detected),
    mask is the binary image used for detection.
    '''
    height, width = z_frame_array.shape
    circular_mask = get_search_mask((height, width), search_rect)

    z_frame = z_frame_array[(z_frame_array > 0) & (circular_mask == 255)]
    if z_frame.size == 0:
        return None, np.zeros((height, width), dtype=np.uint8)

    # Box's top edges are the closest points to the sensor
    min_z = np.min(z_frame)
    depth_range_lower = min_z - 100
    depth_range_upper = min_z + 100
    mask = np.where((z_frame_array >= depth_range_lower) & (z_frame_array <= depth_range_upper), 255, 0).astype(np.uint8)
    mask = cv2.bitwise_and(mask, mask, mask=circular_mask)  # Apply the search mask to remove borders

    # Apply dilation to enhance the box edges
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary mask (where the box is)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_approx = None
    best_area = 0
    for contour in contours:
        # Filter out small unwanted polygon detections
        area = cv2.contourArea(contour)
        if area < 1000:  # Adjust area threshold based on expected box size
            continue

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has 4 vertices, it is likely a rectangle. Keep the largest one.
        if len(approx) == 4 and area > best_area:
            best_approx = approx
            best_area = area

    return best_approx, mask


def get_edge_depth(z_frame_array, approx, box_base_depth, size=10):
    ''' Depth of the box's top edge: median over the four vertices of the minimum depth near each vertex '''
    rows, cols = z_frame_array.shape
    edge_depths = []
    for i in range(4):
        x, y = (int(p) for p in approx[i][0])
        # Get minimum depth value from nearby +/- size indices (clamped to the frame)
        window = z_frame_array[max(y-size, 0):min(y+size, rows), max(x-size, 0):min(x+size, cols)]
        non_zero_window = window[window > 0]
        if non_zero_window.size > 0:
            edge_depths.append(np.min(non_zero_window))
    return float(np.median(edge_depths)) if edge_depths else box_base_depth


# Search area selected with the mouse (x1, y1, x2, y2). None means the default center area.
search_rect = None
drag_start = None


def click_event(event, x, y, flags, param):
    ''' Drag with the left mouse button to select the area where the box is searched for. Right click resets it. '''
    global search_rect, drag_start
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
        x1, x2 = sorted((drag_start[0], x))
        y1, y2 = sorted((drag_start[1], y))
        drag_start = None
        if x2 - x1 > 20 and y2 - y1 > 20:  # Ignore simple clicks
            search_rect = (x1, y1, x2, y2)
            print(f"Search area set to {search_rect}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        search_rect = None
        print("Search area reset to the center of the image")


if __name__ == "__main__":
    #---------------------------------------------camera initialization---------------------------------------------------------------
    serial = input("Enter the sensor serial number: ")
    cam = tof.TuiCamera(serial) #connect to the camera

    names = cam.getMultipleNames() # Get streaming modes information
    print(names)
    cam.setOutputFrameTypes([tof.FrameType.X, tof.FrameType.Y, tof.FrameType.Z]) # Set streaming frame types to X, Y, Z.
    cam.switchConfig(4)

    cam.start() # Start camera streaming

    for _ in range(10):  # As first initial frames from the camera can be inaccurate, drop the first 10 frames
        frame = cam.getFrames()

    # Initializing variables for further use
    box_alignment_set = 0  # Variable to prompt user to finalize the box's position
    approx_box = None   # To store four vertices of the rectangle detected
    reference_frame_box = None   # Reference frame of the empty box to compare with filling in real time
    box_base_depth = 0.0
    box_height = 0.0

    # To show stable width and length
    first_width = None
    first_length = None

    # UI for canvas window
    frame_counter = 0
    # canvas window
    window_name = "Dimensions and fill"
    width, height = 300, 250 #window dimensions
    canvas = np.zeros((height, width, 3), dtype="uint8")

    cv2.namedWindow('Box Filling Application')
    cv2.setMouseCallback('Box Filling Application', click_event)

    print("Place the box under the sensor. Press 'f' to finalize box position, 'n' for a new box, 'q' to quit.")
    print("The box is searched for inside the white area. Drag with the mouse to change it, right click to reset it.")

    while cam.isStreaming():

        frame = cam.getFrames() # capture a frame
        x_frame_array, y_frame_array, z_frame_array = get_xyz_arrays(frame)

        frame_counter += 1 # canvas window
        if frame_counter > 500:
            frame_counter = 0

        # Normalize the depth data for better visualization (scaling from 0 to 255)
        frame_normalized_for_display = cv2.normalize(z_frame_array, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the normalized frame to 8-bit format
        frame_normalized_for_display = np.uint8(frame_normalized_for_display)

        # frame_colormap is of original frame array size
        frame_colormap = cv2.applyColorMap(frame_normalized_for_display, cv2.COLORMAP_JET)

        # Read the keyboard only once per frame so key presses aren't lost
        key = cv2.waitKey(1) & 0xFF

        # ---------------------------------------------contours - detect box ---------------------------------------------------------------------------

        # If the box's position is not set, continously detect the box.
        if box_alignment_set == 0:
            approx, mask = detect_box(z_frame_array, search_rect)
            cv2.imshow("Mask", mask)

            # Draw the search area outline in white
            search_contours, _ = cv2.findContours(get_search_mask(z_frame_array.shape, search_rect),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_colormap, search_contours, -1, (255, 255, 255), 1)

            canvas.fill(0)
            if approx is None:
                cv2.putText(canvas, "No box detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            else:
                for i in range(4):
                    pt1 = tuple(int(p) for p in approx[i][0])
                    pt2 = tuple(int(p) for p in approx[(i+1) % 4][0])
                    cv2.line(frame_colormap, pt1, pt2, (255, 0, 255), 3)  # Pink lines to mark box detected
                cv2.putText(canvas, "Box detected.", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                cv2.putText(canvas, "Press 'f' to finalize", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                # When 'f' is pressed, save approx and the reference frame of the empty box
                if key in (ord('f'), ord('F')):
                    approx_box = copy.deepcopy(approx)
                    reference_frame_box = z_frame_array.copy()
                    box_base_depth = get_center_depth(z_frame_array)  # For getting box's base depth measurement
                    edge_depth = get_edge_depth(z_frame_array, approx_box, box_base_depth)
                    box_height = box_base_depth - edge_depth
                    box_alignment_set = 1
                    print("Box position is set.")
                    print("Reference frame captured.")

        if approx_box is not None:

            canvas.fill(0)

            flashing = (frame_counter) % 2 == 0 # Flashing effect on the streaming window

            if flashing:
                cv2.circle(canvas, (20, 20), 5, (0, 255, 0), -1) # Green dot on canvas window

            # Always draw the "Sensor Streaming" text (Stable)
            cv2.putText(canvas, "Sensor Streaming", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            for i in range(4):
                pt1 = tuple(int(p) for p in approx_box[i][0])
                pt2 = tuple(int(p) for p in approx_box[(i+1) % 4][0])
                cv2.line(frame_colormap, pt1, pt2, (0, 255, 0), 2)  # Green lines to mark box locked

            # Calculate box's rotation angle
            angle = calculate_box_angle(approx_box)
            cv2.putText(canvas, f"Angle: {angle:.2f} degrees", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # Calculate width and length from the sensor's real-world X, Y, Z frames
            if first_width is None and first_length is None:
                first_width, first_length = calculate_box_dimensions(approx_box, reference_frame_box,
                                                                     x_frame_array, y_frame_array)

            if first_width and first_length:
                cv2.putText(canvas, f"Width: {first_width:.2f} mm", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                cv2.putText(canvas, f"Length: {first_length:.2f} mm", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(canvas, f"Height: {box_height:.2f} mm", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            if box_height <= 0:
                cv2.putText(canvas, "Box height not valid", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            else:
                fill_percentage = calculate_fill_percentage(approx_box, z_frame_array, reference_frame_box, box_height, box_base_depth)
                if fill_percentage == "Not Allowed":
                    cv2.putText(canvas, "Objects are above the box edges", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                else:
                    cv2.putText(canvas, f"Fill: {fill_percentage:.2f}%", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # Press n to detect a new box.
            if key in (ord('n'), ord('N')):
                # Reset all variables related to box detection
                box_alignment_set = 0
                approx_box = None
                reference_frame_box = None
                first_width = None
                first_length = None
                print("Resetting for new box.")

        #------------------------------------------------Real-time stream display----------------------------------------------------------------------
        cv2.circle(frame_colormap, (z_frame_array.shape[1] // 2, z_frame_array.shape[0] // 2), 3, (0, 255, 0), -1)  # Green dot: must fall on the box's base
        cv2.imshow('Box Filling Application', frame_colormap)
        cv2.imshow(window_name, canvas)

        # Break the loop if 'q' is pressed
        if key in (ord('q'), ord('Q')):
            break

    # Stop streaming and disconnect the camera
    cam.stop()
    cam = None
    cv2.destroyAllWindows()
