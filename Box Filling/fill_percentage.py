import numpy as np
import cv2

def calculate_fill_percentage(approx, z_frame_array, reference_frame, box_height, box_base_depth):
    """
    Calculate volume fill percentage of a box by comparing the current depth frame to the reference frame.
    
    Parameters:
    1. approx: Detected four vertices of the box.
    2. z_frame_array: The current depth frame with objects inside the box.
    3. reference_frame: The reference depth frame of the empty box.
    4. box_height: The height of the box.
    5. box_base_depth: The depth value at the base of the box.
    
    Returns:
    The fill percentage of the box based on the volume filled.
    """
    
    # Ensure approx is converted to a NumPy array with integer coordinates
    polygon = np.array(approx, dtype=np.int32)
    
    # Reshape to the required format for cv2.fillPoly (shape: (1, n, 2))
    polygon = polygon.reshape((-1, 1, 2))

    # Create an empty mask with the same size as the depth frame
    mask = np.zeros_like(z_frame_array, dtype=np.uint8)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)

    # Use cv2.fillPoly to fill the area inside the polygon (box)
    cv2.fillPoly(mask, [polygon], color=255)
    # cv2.imshow("Mask", mask)

    # Extract the depth values inside the box for both current and reference frames
    current_box_depths = z_frame_array[mask == 255]
    reference_box_depths = reference_frame[mask == 255]

    valid_current_frame = current_box_depths[current_box_depths <= box_base_depth+50]
    valid_reference_frame = reference_box_depths[reference_box_depths <= box_base_depth+50]

    box_top_depth = box_base_depth - box_height

    valid_current_frame = valid_current_frame[valid_current_frame != 0]
    if np.any(valid_current_frame <= box_top_depth - 50):
        return "Not Allowed"

    else:
        # Calculate how much space is "empty" in both frames (closer to base means more empty)
        reference_empty_space = (valid_reference_frame - box_top_depth) / box_height
        current_empty_space = (valid_current_frame - box_top_depth) / box_height

        # Calculate the total space available and the space currently filled
        total_empty_space = np.sum(reference_empty_space)  # Total space to fill (from reference)
        current_empty_space_sum = np.sum(current_empty_space)  # Remaining empty space (from current)

        # Calculate how much has been filled
        filled_volume = total_empty_space - current_empty_space_sum

        # If no significant fill, box is considered empty
        if filled_volume <= 0:
            return 0

        # Calculate the percentage filled
        fill_percentage = (filled_volume / total_empty_space) * 100

    return fill_percentage
