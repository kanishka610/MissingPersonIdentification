import cv2
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Load the video file
video_path = 'Birthday.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

while True:
    # Read each frame from the video
    ret, frame = cap.read()

    # Break the loop if no frames are left
    if not ret:
        break

    # Detect faces in the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    # Draw rectangles around detected faces
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Video', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
