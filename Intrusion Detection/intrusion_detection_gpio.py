import os
os.add_dll_directory(r"C:\Program Files\tof")
os.add_dll_directory(r"C:\Program Files\tof\bin") 
import cv2
import numpy as np
import chronoptics.tof as tof

# Initialize 3D camera
serial = input("Enter sensor's serial number: ")
cam = tof.TuiCamera(serial=serial)

cam.setOutputFrameTypes([tof.FrameType.Z])

# Configure GPIO Mode 
cam.setTriggerMode(tof.TriggerMode.CONTINUOUS)
cam.setGpioMode(0, tof.GpioMode.GENERAL_INPUT)
cam.setGpioMode(1, tof.GpioMode.GENERAL_OUTPUT)
cam.setGpioMode(2, tof.GpioMode.GENERAL_OUTPUT)
# Initializing output pins to low state
cam.setGpio(1, 1)
cam.setGpio(2, 1)

cam.start()

cam.switchConfig(6)

# Global variables for the intrusion detection system
draw = False  # True if the mouse is pressed. Used for selecting ROI.
roi_selected = False  # Flag to check if ROI has been selected
rect = (0, 0, 0, 0)  # Coordinates of the rectangle (ROI)
start_point = (0, 0)  # Starting point of the rectangle

MIN_SIZE_THRESHOLD = 50
TEMPORAL_THRESHOLD = 3  # Number of consecutive frames to confirm intrusion
depth_threshold_min = 0
depth_threshold_max = 0

# Variable for depicting sum of intrusions in consecutive frames
consecutive_intrusions = 0

is_distance_set = False


def click_event(event, x, y, flags, param):
    global start_point, draw, rect, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        draw = True
        roi_selected = False
    elif event == cv2.EVENT_MOUSEMOVE and draw:
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        roi_selected = True
        rect = (start_point[0], start_point[1], x, y)

# Function to check for intrusion in the selected ROI and depth range
def check_intrusion(depth_frame, rect, depth_threshold, min_size_threshold):
    # Crop the depth frame to the selected ROI
    x, y, x2, y2 = rect
    cropped_depth = depth_frame[y:y2, x:x2]
    
    intrusion_mask = (cropped_depth > depth_threshold[0]) & (cropped_depth < depth_threshold[1])
    intrusion_size = np.sum(intrusion_mask)

    intrusion_detected = intrusion_size > min_size_threshold
    return intrusion_detected, intrusion_size

# Function to highlight text with background
def highlight_text(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, text_color=(0, 0, 255), highlight_color=(0, 255, 255)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position
    
    # Draw a filled rectangle behind the text (highlight color)
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), highlight_color, thickness=cv2.FILLED)
    
    # Draw the text on top of the rectangle (text color)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_event)


while cam.isStreaming():
    frames = cam.getFrames()
    depth_frame = np.asarray(frames[0], dtype=np.float32)
    depth_frame = np.rot90(depth_frame, -1)
    # depth converted to a visual format for display
    depth_visual = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = depth_visual.astype(np.uint8)  # Ensure the image is in 8-bit format
    # Color map applied to the normalized depth image
    colored_depth = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    # Colored depth image for visualization and further processing
    frame = colored_depth

    cv2.imshow("Frame", frame)

    if not is_distance_set:
            if depth_threshold_min == 0 and depth_threshold_max == 0:
                depth_threshold_min = int(input("Please input minimum depth in mm, range 400 - 15000, default 1000 : ") or 1000)
                depth_threshold_max = int(input("Please input maximum depth in mm, range 400 - 15000, default 3000 : ") or 3000)
                DEPTH_THRESHOLD = (depth_threshold_min, depth_threshold_max)  # depth threshold in mm# Mouse callback function for selecting ROI
                is_distance_set = True


    if roi_selected:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        if is_distance_set:    
            intrusion_detected, intrusion_size = check_intrusion(depth_frame, rect, DEPTH_THRESHOLD, MIN_SIZE_THRESHOLD)

            if intrusion_detected:
                consecutive_intrusions += 1
            else:
                consecutive_intrusions = 0
                cam.setGpio(1, 1)

            if consecutive_intrusions >= TEMPORAL_THRESHOLD:
                # If intrusion is detected, set output pin 1 to high
                cam.setGpio(1, 0)

                # If intrusion is observed in frames equal to or more than temporal threshold defined, then display the text as intrusion detected
                # Highlight "Intrusion Detected!" text in red with yellow background.
                highlight_text(frame, "Intrusion Detected!", (10, 30), font_scale=1, text_color=(0, 0, 255), highlight_color=(0, 255, 255))
                # Draw a red rectangle around the detected intrusion region
                cv2.rectangle(frame, (rect[0]+10, rect[1]+10), (rect[2]+10, rect[3]+10), (0, 0, 255), 2)  # Red color for intrusion area
        

    cv2.imshow("Frame", frame)

    # Break the loop with the 'q' key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # Cleanup

    if cv2.waitKey(1) & 0xFF == ord('n'):
        roi_selected = False
        is_distance_set = False
        depth_threshold_min = 0
        depth_threshold_max = 0


cam.stop()
cv2.destroyAllWindows()
cam = None
