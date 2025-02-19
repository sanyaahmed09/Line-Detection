import cv2
import numpy as np
def detect_parallel_lines(frame):
 """Detects and overlays curved parallel lines and their centerline."""
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 blurred = cv2.GaussianBlur(gray, (5, 5), 0)

 edges = cv2.Canny(blurred, 50, 150)

 # Find contours
 contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 # Sort contours by size and filter small ones
 contours = sorted(contours, key=cv2.contourArea, reverse=True)
 valid_lines = [cnt for cnt in contours if cv2.contourArea(cnt) > 500] # Adjust
threshold if needed

 # Draw detected curved parallel lines (Green)
 for line in valid_lines:
 cv2.drawContours(frame, [line], -1, (0, 255, 0), 2)

 # Find and draw centerline
 if len(valid_lines) >= 2:
 line1 = valid_lines[0].reshape(-1, 2) # Convert to (x, y) points
 line2 = valid_lines[1].reshape(-1, 2)

 # Ensure both lines have the same number of points by interpolation
 min_len = min(len(line1), len(line2))
 line1 = line1[:min_len]
 line2 = line2[:min_len]

 centerline = (line1 + line2) // 2 # Compute centerline (average points)

 # Draw centerline
 cv2.polylines(frame, [centerline], isClosed=False, color=(0, 0, 255),
thickness=2)
 return frame
# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
 ret, frame = cap.read()
 if not ret:
 break

 processed_frame = detect_parallel_lines(frame)

 cv2.imshow("Curved Line Detection", processed_frame)

 if cv2.waitKey(1) & 0xFF == ord('q'):
 break
cap.release()
cv2.destroyAllWindows()
