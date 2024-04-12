import cv2
import mediapipe as mp

def resize_frame(frame, width=None, height=None):
    if width is not None and height is not None:
        return cv2.resize(frame, (width, height))
    elif width is not None:
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_height = int(width / aspect_ratio)
        return cv2.resize(frame, (width, new_height))
    elif height is not None:
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = int(height * aspect_ratio)
        return cv2.resize(frame, (new_width, height))
    else:
        return frame

def main():
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open video file
    video_path = 'data/male-bodyweight-hand-plank.mp4'  # Reemplaza 'tu_video.avi' con la ruta de tu archivo de video
    cap = cv2.VideoCapture(video_path)

    # Especifica el ancho y alto deseados para el video redimensionado
    new_width = 800# Ancho deseado para el video redimensionado (píxeles)
    new_height = 600  # Alto deseado para el video redimensionado (píxeles)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame
        frame_resized = resize_frame(frame, new_width, new_height)

        # Convertir el frame a RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Realizar la detección de pose
        results = pose.process(frame_rgb)

        # Dibujar los landmarks de pose en el frame
        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mostrar el frame redimensionado
        cv2.imshow('Pose Estimation', frame_resized)

        # Comprobar la tecla de salida
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
