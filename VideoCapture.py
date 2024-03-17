import cv2
import numpy as np

# Open the video file for capture
cap = cv2.VideoCapture('vtest.avi')

# Read the first two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Print shapes of the frames
print("Frame 1 shape:", frame1.shape)
print("Frame 2 shape:", frame2.shape)

# Main loop to process each frame
while cap.isOpened():
    # Calculate the absolute difference between two consecutive frames
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BAYER_BGGR2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to the blurred image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill gaps in between object edges
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours of objects in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original frame
    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    
    # Display the frame with drawn contours
    cv2.imshow("feed", frame1)
    
    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = cap.read()
    
    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(40) == 27:
        break

# Release video capture and close all windows
cv2.destroyAllWindows()
cap.release()
