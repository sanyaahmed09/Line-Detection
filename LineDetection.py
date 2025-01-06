import cv2
import numpy as np

def detect_lines(frame):
    # converts into greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # applies Canny edge detection (highlights areas with strong contrasts)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return frame, None

    detected_lines = []
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        detected_lines.append(((x1, y1), (x2, y2), rho, theta))
    
    # Filter for parallel lines based on their angles
    parallel_lines = []
    for i, line1 in enumerate(detected_lines):
        for j, line2 in enumerate(detected_lines):
            if i != j and abs(line1[3] - line2[3]) < 0.1:  # Angle difference threshold
                parallel_lines = [line1, line2]
                break
        if parallel_lines:
            break

    if len(parallel_lines) == 2:
        # Draw the parallel lines
        for line in parallel_lines:
            cv2.line(frame, line[0], line[1], (0, 255, 0), 2)

        # Calculate the center line
        x1_mid = (parallel_lines[0][0][0] + parallel_lines[1][0][0]) // 2
        y1_mid = (parallel_lines[0][0][1] + parallel_lines[1][0][1]) // 2
        x2_mid = (parallel_lines[0][1][0] + parallel_lines[1][1][0]) // 2
        y2_mid = (parallel_lines[0][1][1] + parallel_lines[1][1][1]) // 2

        # draw the center line
        cv2.line(frame, (x1_mid, y1_mid), (x2_mid, y2_mid), (255, 0, 0), 2)

        return frame, ((x1_mid, y1_mid), (x2_mid, y2_mid))

    return frame, None


def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture the frame.")
            break

        # Process each frame
        processed_frame, centerline = detect_lines(frame)

        # show the result
        cv2.imshow("Parallel Line Detection", processed_frame)

        # stop when you press f
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Sources:
https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
https://www.youtube.com/watch?v=gbL3XKOiBvw&ab_channel=ProgrammingKnowledge
"""
