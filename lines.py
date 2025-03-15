import cv2
import numpy as np

def detect_lines(frame):
    height, width = frame.shape[:2]
    
    # Define a triangular ROI (adjustable)
    triangle_vertices = np.array([
        (int(width * 0.20), height),  # Bottom-left
        (int(width * 0.85), height),  # Bottom-right
        (int(width * 0.5), int(height * 0.75))  # Top (apex)
    ], np.int32)
    
    # Create mask for the triangular ROI
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [triangle_vertices], (255, 255, 255))
    roi = cv2.bitwise_and(frame, mask)
    
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Hough Transform to detect lane lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    left_lines, right_lines = [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.3:  # Left lane
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:  # Right lane
                right_lines.append((x1, y1, x2, y2))
    
    # Refine the lines: Take the average line for both left and right lanes
    def average_line(lines):
        if not lines:
            return None
        avg_x1 = np.mean([line[0] for line in lines])
        avg_y1 = np.mean([line[1] for line in lines])
        avg_x2 = np.mean([line[2] for line in lines])
        avg_y2 = np.mean([line[3] for line in lines])
        return int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)
    
    left_line = average_line(left_lines)
    right_line = average_line(right_lines)
    
    # Draw lane lines with red holographic effect
    holographic_color = (0, 0, 255)  # Red color for holographic effect
    holographic_alpha = 0.4  # Transparency level for holographic effect

    def draw_holographic_lines(lines, frame):
        for line in lines:
            x1, y1, x2, y2 = line
            overlay = np.zeros_like(frame, dtype=np.uint8)  # Create a blank overlay
            cv2.line(overlay, (x1, y1), (x2, y2), holographic_color, 3)  # Draw red line on the overlay
            # Blend the overlay with the frame to get the holographic effect
            frame = cv2.addWeighted(frame, 1 - holographic_alpha, overlay, holographic_alpha, 0)
        return frame

    if left_line:
        frame = draw_holographic_lines([left_line], frame)
    
    if right_line:
        frame = draw_holographic_lines([right_line], frame)

    # Compute and draw midline
    if left_line and right_line:
        left_x = np.mean([left_line[0], left_line[2]])
        right_x = np.mean([right_line[0], right_line[2]])
        mid_x = int((left_x + right_x) / 2)
        cv2.line(frame, (mid_x, height), (mid_x, int(height * 0.5)), (255, 0, 0), 2)  # Midline in blue
    
    return frame, edges  # Return both processed frame and edge-detection result

def overlay_image(background, overlay, x, y):
    """Overlays a transparent image (overlay) onto a background at (x, y)."""
    overlay_h, overlay_w = overlay.shape[:2]

    # Ensure overlay fits in the background frame
    if x + overlay_w > background.shape[1] or y + overlay_h > background.shape[0]:
        return background

    # Extract the alpha channel (transparency mask)
    b, g, r, a = cv2.split(overlay)

    # Normalize alpha to range 0-1
    alpha = a / 255.0

    # Select region of interest (ROI) on the background
    roi = background[y:y+overlay_h, x:x+overlay_w]

    # Blend images using the alpha mask
    for c in range(3):  # Loop through color channels (B, G, R)
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay[:, :, c]

    # Put blended ROI back into the original frame
    background[y:y+overlay_h, x:x+overlay_w] = roi
    return background

# video and image paths
input_video_path = "/Users/profasarwat/Desktop/Video Detection Sanya (1).MP4"
arrow_image_path = "/Users/profasarwat/Desktop/arrow-png-image.png"  # Ensure this is a transparent PNG

# Load the arrow image
arrow = cv2.imread(arrow_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

if arrow is None:
    print("Error: Could not load arrow image.")
    exit()

# Resize the arrow if needed
arrow = cv2.resize(arrow, (80, 80))  # Adjust size as needed

# Open video file
video = cv2.VideoCapture(input_video_path)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  
fps = video.get(cv2.CAP_PROP_FPS)  

output_video_path = "output_video.avi" 
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Process the frame
    processed_frame, edges = detect_lines(frame)

    # Overlay the arrow in the top-left corner
    processed_frame = overlay_image(processed_frame, arrow, 20, 20)  # Adjust position as needed

    # Display the processed frame (overlay with arrow)
    cv2.imshow("Overlay Video", processed_frame)

    # Display the frame with detected edges (line detection)
    cv2.imshow("Line Detection", edges)

    # Write the processed frame to the output video
    out.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()  # Release the video object
out.release()  # Release the output video object
cv2.destroyAllWindows()  # Close any OpenCV windows
