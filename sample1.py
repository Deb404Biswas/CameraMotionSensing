import cv2
import numpy as np

# Set up video capture
cap = cv2.VideoCapture(0)

# Set up background subtraction
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Threshold the foreground mask
    thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]

    # Remove noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Perform color correction
    frame_corrected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_corrected = cv2.cvtColor(frame_corrected, cv2.COLOR_RGB2BGR)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around any moving objects
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame_corrected, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame_corrected)

    # Check for motion and send an alert
    if len(contours) > 0:
        print("Motion detected!")

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
