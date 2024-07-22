import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
import threading

# Initialize pygame mixer
pygame.mixer.init()

# Load the pre-trained face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define constants
EYE_AR_THRESH = 0.25  # Adjusted threshold for EAR
EYE_AR_CONSEC_FRAMES = 48  # Adjusted number of frames
ALARM_FILE = 'alert.wav'

# Initialize the frame counters and the total number of blinks
COUNTER = 0
ALARM_ON = False

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to sound the alarm in a separate thread
def sound_alarm():
    pygame.mixer.music.load(ALARM_FILE)
    pygame.mixer.music.play()

# Start the video stream
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Extract the coordinates of the left and right eye
        leftEye = shape[36:42]
        rightEye = shape[42:48]

        # Compute the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Take the average of both eye aspect ratios
        ear = (leftEAR + rightEAR) / 2.0

        # Visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the blink threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # If the eyes have been closed for a sufficient number of frames, sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    threading.Thread(target=sound_alarm).start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False
            pygame.mixer.music.stop()

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
