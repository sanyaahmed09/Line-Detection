import cv2
import numpy as np

def detect_lines(frame):
    # define ROI box
    height, width = frame.shape[:2]
    roi_top_left = (int(width * 0.20), int(height * 0.75))
    roi_bottom_right = (int(width * 0.7), int(height * 1.0))

    # Draw the ROI box; in color yellow
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 255), 2)

    # use ROI for line detection; only detect what's in there
    roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # turn into binary and blur
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)

    # Edge detection using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Detect contours (to better isolate thick lines)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        # Filter small noise
        if cv2.contourArea(contour) > 500:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vy / (vx + 1e-6)
            lines.append((vx, vy, x, y, slope))

    if len(lines) >= 2:
        # Sort lines by horizontal position (x) or vertical position (y) dynamically
        lines = sorted(lines, key=lambda l: (l[2] if abs(l[4]) < 1 else l[3]))  # Use x for slanted, y for vertical

        # Select two most prominent lines
        line1 = lines[0]
        line2 = lines[1]

        # Draw detected lines in the ROI
        for vx, vy, x, y, slope in [line1, line2]:
            x1, y1 = int(x - 1000 * vx) + roi_top_left[0], int(y - 1000 * vy) + roi_top_left[1]
            x2, y2 = int(x + 1000 * vx) + roi_top_left[0], int(y + 1000 * vy) + roi_top_left[1]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        # Compute centerline based on line orientations
        if abs(line1[4]) > 1 and abs(line2[4]) > 1:  # Mostly vertical lines
            # Average x-coordinates for vertical centerline
            center_x = int((line1[2] + line2[2]) // 2) + roi_top_left[0]
            midline = [(center_x, roi_top_left[1]), (center_x, roi_bottom_right[1])]
        else:  # Slanted or horizontal lines
            # Average y-coordinates for horizontal centerline
            center_y = int((line1[3] + line2[3]) // 2) + roi_top_left[1]
            midline = [(roi_top_left[0], center_y), (roi_bottom_right[0], center_y)]

        # Draw the centerline
        cv2.line(frame, tuple(map(int, midline[0])), tuple(map(int, midline[1])), (255, 0, 0), 2)  # Blue for centerline

    return frame


# Video processing code
input_video_path = "/Users/pl1004218/Desktop/video_detection.MP4"
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
    processed_frame = detect_lines(frame)

    # Display the processed frame
    cv2.imshow("Video Frame with Line Detection", processed_frame)

    # Write the processed frame to the output video
    out.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()  # Release the video object
out.release()  # Release the output video object
cv2.destroyAllWindows()  # Close any OpenCV windows
