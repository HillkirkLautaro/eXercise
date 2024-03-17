import cv2
import mediapipe as mp

# Open the video file for capture
cap = cv2.VideoCapture(r'data/Salto Tijera (720p).mp4')


def main():
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open the video file for capture
    cap = cv2.VideoCapture(r'C:\Users\123la\Documents\GitHub\repositorios de github\EXERCISE\data\Salto Tijera (720p).mp4')


    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Pose Estimation', frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()