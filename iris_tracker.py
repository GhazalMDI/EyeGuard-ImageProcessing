from email.mime import image

import cv2
import mediapipe as mp
from matplotlib.pyplot import annotate
from mediapipe.calculators import video

mpDrawing = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mpDrawingStyles = mp.solutions.drawing_styles
drawSpecs = mpDrawingStyles.DrawingSpec(thickness=1, circle_radius=1)

def get_landmarks(image):
    faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True, max_num_faces=1,refine_landmarks=True, min_detection_confidence=0.5)
    image.flags.writeable = False
    result=faceMesh.process(image)
    landmark=result.multi_face_landmarks[0].landmark
    return result,landmark

def draw_landmarks(image, result):
    image.flags.writeable = True
    if result.multi_face_landmarks:
        for faceLandmark in result.multi_face_landmarks:
            mpDrawing.draw_landmarks(
                image,
                faceLandmark,
                connections=mpFaceMesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_iris_connections_style()
            )

    return image

cap=cv2.VideoCapture(0)
cap.set(10,100) #brightness
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_cp=cv2.flip(frame,1)
    result,landmark=get_landmarks(frame_cp)
    annotate_frame = draw_landmarks(frame_cp, result)
    cv2.imshow("frame", frame_cp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
