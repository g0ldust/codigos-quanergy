import chronoptics.tof as tof
import numpy as np
import copy
import cv2
from collections import deque
from box_measurements import calculate_box_dimensions
from fill_percentage import calculate_fill_percentage
from box_angle import calculate_box_angle

#---------------------------------------------camera initialization---------------------------------------------------------------
serial = input("Enter the sensor serial number: ")
cam = tof.TuiCamera(serial) #connect to the camera

# get calibration to calculate box dimensions
calibration = cam.getCalibration()
focal_x = calibration.getFocalLengthX()
focal_y = calibration.getFocalLengthY()
principal_p_x = calibration.getPrincipalPointX()
principal_p_y = calibration.getPrincipalPointY()


names = cam.getMultipleNames() # Get streaming modes information
print(names)
cam.setOutputFrameTypes([tof.FrameType.X, tof.FrameType.Y, tof.FrameType.Z]) # Set streaming frame type to Z.
cam.switchConfig(4) 

cam.start() # Start camera streaming

for _ in range(10):  # As first initial frames from the camera can be inaccurate, drop the first 10 frames
    frame = cam.getFrames()

# For getting box's base depth measurement
z_frame_array = np.asarray(frame[2])
z_frame_array = np.rot90(z_frame_array, -1)
row_index = z_frame_array.shape[0] // 2
col_index = z_frame_array.shape[1] // 2
floor_depth = z_frame_array[row_index][col_index]

# Initializing variables for further use
box_alignment_set = 0  # Variable to prompt user to finalize the box's position
''' Variable to store reference frame to comapre it with filling in real time to 
get volume filled '''
reference_frame = np.zeros_like(z_frame_array)   # Frame of the same shape as z depth frame
approx_box = None   # To store four vertices of the rectangle detected

# To show stable width and length
first_width = None
first_length = None

# UI for canvas window
frame_counter = 0
# canvas window
window_name = "Dimensions and fill"
width, height = 300, 250 #window dimensions
canvas = np.zeros((height, width, 3), dtype="uint8")

while cam.isStreaming(): 

    frame = cam.getFrames() # capture a frame

    z_frame_array = np.asarray(frame[2])   # Convert to array for display and further operations
    z_frame_array = np.rot90(z_frame_array, -1)

    x_frame_array = np.asarray(frame[0])
    x_frame_array = np.rot90(x_frame_array, -1)

    y_frame_array = np.asarray(frame[1])
    y_frame_array = np.rot90(y_frame_array, -1)

    frame_counter += 1 # canvas window

    # # Display the updated canvas
    # cv2.imshow("Dimensions and fill", canvas)

    if frame_counter > 500:
        frame_counter = 0

    contour_frame = np.zeros_like(z_frame_array)

    # Normalize the depth data for better visualization (scaling from 0 to 255)
    frame_normalized_for_display = cv2.normalize(z_frame_array, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized frame to 8-bit format
    frame_normalized_for_display = np.uint8(frame_normalized_for_display)

    # frame_colormap is of original frame array size
    frame_colormap = cv2.applyColorMap(frame_normalized_for_display, cv2.COLORMAP_JET)

    # ---------------------------------------------contours - detect box ---------------------------------------------------------------------------
    
    # If the box's position is not set, continously detect the box and capture the approx.
    # If the box position is set then box's position is detected in real time for getting approx.
    if box_alignment_set == 0:
        # Find contours of the box in the depth image
        height, width = z_frame_array.shape
        # Create a circular mask (set the circle to 255, rest to 0)
        circular_mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)  # Center of the image
        radius = min(center[0], center[1])  # Radius for the circle to cover the round lens
        cv2.circle(circular_mask, center, radius, 255, -1)  # Draw a filled circle

        # Apply the circular mask to remove the border
        z_frame = z_frame_array[z_frame_array > 0]
        min_z = np.min(z_frame)
        depth_range_lower = min_z - 100
        depth_range_upper = min_z + 100
        mask = np.where((z_frame_array >= depth_range_lower) & (z_frame_array <= depth_range_upper), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_and(mask, mask, mask=circular_mask)  # Apply the circular mask to remove borders

        # Apply dilation to enhance the box edges
        kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Mask", mask)

        

        ### --- Step 2: Apply Contour Detection --- ###
        # Find contours in the binary mask (where the box is)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        edge_depth = 0 # Box's edge depth variable

        if len(contours) == 0:
            print("No box detected")
            continue

        ### --- Step 4: Draw Green Rectangles on the Colormap --- ###
        # Iterate through contours and filter for rectangular shapes
        for contour in contours:
            # Filter out small unwanted polygon detections
            area = cv2.contourArea(contour)
            if area < 1000:  # Adjust area threshold based on expected box size
                continue
            
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the polygon has 4 vertices, it is likely a rectangle
            if len(approx) == 4:
                # For getting box's base depth measurement
                z_frame_array = np.asarray(frame[2])
                z_frame_array = np.rot90(z_frame_array, -1)
                row_index = z_frame_array.shape[0] // 2
                col_index = z_frame_array.shape[1] // 2
                box_base_depth = z_frame_array[row_index][col_index]

                for i in range(4):
                    vertex = tuple(approx[i][0])
                    x, y = vertex

                    # Logic for calculating box height
                    # Get minimum depth value from nearby +/- 10 indices
                    window = z_frame_array[y-10:y+10, x-10:x+10]
                    # Mask out zero values and find the minimum of the non-zero elements
                    non_zero_window = window[window > 0]
                    if non_zero_window.size > 0:
                        edge_depth = np.min(non_zero_window)
                    else:
                        edge_depth = box_base_depth  # Fallback to base depth if all values are zero
                    # print(edge_depth)
                
                    pt1 = tuple(approx[i][0])
                    pt2 = tuple(approx[(i+1) % 4][0])
                    cv2.line(frame_colormap, pt1, pt2, (255, 0, 255), 3)  # Pink lines to mark box detected
                    # cv2.putText(frame_colormap, "Press 'f' to finalize box alignment", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # When 'f' is pressed, save approx
                    if cv2.waitKey(1) & 0xFF == ord('f') or cv2.waitKey(1) & 0xFF == ord('F'):
                        approx_box = copy.deepcopy(approx)
                        reference_frame_box = copy.deepcopy(z_frame_array)
                        box_alignment_set = 1
                        print("Box position is set.")
                        print("Reference frame captured.")
    

    
    if approx_box is not None:

        # # Define colors for each vertex (BGR format)
        # colors = [(0, 0, 255),  # Red for the 1st point
        #       (0, 255, 0),  # Green for the 2nd point
        #       (255, 0, 0),  # Blue for the 3rd point
        #       (0, 255, 255)]  # Yellow for the 4th point
        

        # for i in range(4):
        #     vertex = tuple(approx_box[i][0])  # Extract the (x, y) coordinates of the vertex
        #     cv2.circle(frame_colormap, vertex, 5, colors[i], -1)

        # Calculate box's rotation angle


        canvas.fill(0)

        flashing = (frame_counter) % 2 == 0 # Flashing effect on the streaming window

        if flashing:
            cv2.circle(canvas, (20, 20), 5, (0, 255, 0), -1) # Green dot on canvas window
        
        # Always draw the "Sensor Streaming" text (Stable)
        cv2.putText(canvas, "Sensor Streaming", (35, 25), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))

        angle = calculate_box_angle(approx_box)

        # cv2.putText(frame_colormap, f"Angle: {angle:.2f} degrees", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Angle: {angle:.2f} degrees", (10, 80), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))
        # Calculate width and length based 
        width, length = calculate_box_dimensions(approx_box, z_frame_array, x_frame_array, y_frame_array, 
                                                 focal_x,
                                                 focal_y,
                                                 principal_p_x,
                                                 principal_p_y)
        if width and length:
            if first_width is None and first_length is None:
                first_width = width
                first_length = length

            # cv2.putText(frame_colormap, f"Width: {width:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # cv2.putText(frame_colormap, f"Length: {length:.2f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, f"Width: {first_width:.2f} mm", (10, 110), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))
            cv2.putText(canvas, f"Length: {first_length:.2f} mm", (10, 140), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))
        box_height = box_base_depth - edge_depth
        # cv2.putText(frame_colormap, f"Height: {box_height:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Height: {box_height:.2f} mm", (10, 170), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))


        fill_percentage = calculate_fill_percentage(approx_box, z_frame_array, reference_frame_box, box_height, box_base_depth)
        if fill_percentage == "Not Allowed":
            # cv2.putText(frame_colormap, "Objects are above the box edges", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, "Objects are above the box edges", (10, 200), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))

        else:
            # cv2.putText(frame_colormap, f"Fill: {fill_percentage:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, f"Fill: {fill_percentage:.2f}%", (10, 200), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))


        # Press n to detect a new box.
        if cv2.waitKey(1) & 0xFF == ord('n') or cv2.waitKey(1) & 0xFF == ord('N'):
            # Reset all variables related to box detection
            box_alignment_set = 0
            approx_box = None
            reference_frame_box = reference_frame = np.zeros_like(z_frame_array)
            first_width = None
            first_length = None
            print("Resetting for new box.")

           
    
    #------------------------------------------------Real-time stream display----------------------------------------------------------------------
    # Frame display functions 
    # cv2.imshow('Box Mask (Binary)', mask)
    cv2.circle(frame_colormap, (z_frame_array.shape[1] // 2, z_frame_array.shape[0] // 2), 3, (0, 255, 0), -1)  # Red dot with radius 5
    cv2.imshow('Box Filling Application', frame_colormap)
    cv2.imshow(window_name, canvas)

    # Wait for a short period and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
        break

# Stop streaming and disconnect the camera
cam.stop()
cam = None
cv2.destroyAllWindows()