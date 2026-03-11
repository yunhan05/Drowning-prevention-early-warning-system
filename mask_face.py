import cv2
import mediapipe as mp

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)


def mask_face(frame):
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame)
    frame_h, frame_w = frame.shape[:2]

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.detections:
        for detection in results.detections:
            face_box = detection.location_data.relative_bounding_box
            xmin, ymin, face_w, face_h = face_box.xmin, face_box.ymin, face_box.width, face_box.height
            l, t = int(xmin*frame_w), int(ymin*frame_h)
            r, b = l+int(face_w*frame_w), t+int(face_h*frame_h)

            cv2.rectangle(frame, (l, t), (r, b), (203, 192, 255), -1)

    return frame
