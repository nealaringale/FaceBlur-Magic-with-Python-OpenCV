# Neal Aringale 

import cv2

# Step 2: Defining the Blur Faces Function
def blur_faces(frame, faces):
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 0)
        frame[y:y+h, x:x+w] = blurred_face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

# Step 3: Initializing Webcam and Face Detection Model
video_capture = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detected_faces = []

# Step 4: Real-Time Face Blurring
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected_faces = faces

    frame = blur_faces(frame, detected_faces)

    cv2.imshow("Blurred Faces", frame)

    if cv2.waitKey(1) != -1:
        break

# Step 5: Cleanup
video_capture.release()
cv2.destroyAllWindows()
