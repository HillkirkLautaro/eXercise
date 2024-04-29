import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the video
video = cv2.VideoCapture(r'data\chinup-front.mp4')

# Check if video opened successfully
if not video.isOpened(): 
    print("Error opening video file")

while(video.isOpened()):
    # Capture frame-by-frame
    ret, frame = video.read()
    
    if ret:
        # Process the image to find pose landmarks
        result = pose.process(frame)

        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            for landmark in result.pose_landmarks.landmark:
                # Get the coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                # Draw a small circle at each landmark
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the resulting frame
        cv2.imshow('Video', frame)
    
    else:
        # Si no hay más frames, reinicia el video
        video.release()
        video = cv2.VideoCapture(r'data\chinup-front.mp4')

    # Quita si se presiona 'q', espera 1 ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break          

# Close all the frames
cv2.destroyAllWindows() 

