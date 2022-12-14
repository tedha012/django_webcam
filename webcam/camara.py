from imutils.video import VideoStream
from keras.models import load_model
import cv2
from PIL import Image
import numpy as np

gender_model = load_model("./data/gender_InceptionResNetV2.hdf5")
age_model = load_model("./data/age_ResNet152V2.hdf5")
detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")


def detect_face(img):

    mt_res = detector.detectMultiScale(img)
    return_res = []

    for face in mt_res:
        x, y, width, height = face
        center = [x + (width / 2), y + (height / 2)]
        max_border = max(width, height)

        # center alignment
        left = max(int(center[0] - (max_border / 2)), 0)
        right = max(int(center[0] + (max_border / 2)), 0)
        top = max(int(center[1] - (max_border / 2)), 0)
        bottom = max(int(center[1] + (max_border / 2)), 0)

        # crop the face
        center_img_k = img[top : top + max_border, left : left + max_border, :]
        center_img = np.array(Image.fromarray(center_img_k).resize([256, 256]))

        # gender prediction
        gender_preds = gender_model.predict(
            np.expand_dims(center_img / 255, 0),
            verbose=0,
        )[0][0]
        if gender_preds > 0.5:
            gender_text = "Female"
        else:
            gender_text = "Male"

        # age prediction
        age_preds = age_model.predict(
            np.expand_dims(center_img / 255, 0),
            verbose=0,
        )
        max_age_preds = np.array(max(age_preds[0]))
        idx_max_age_preds = np.where(age_preds == max_age_preds)[1][0]
        if idx_max_age_preds == 0:
            age_text = "lt 20"
        elif idx_max_age_preds == 1:
            age_text = "20"
        elif idx_max_age_preds == 2:
            age_text = "30"
        elif idx_max_age_preds == 3:
            age_text = "40"
        elif idx_max_age_preds == 4:
            age_text = "50"
        elif idx_max_age_preds == 5:
            age_text = "gt 60"
        else:
            age_text = "NONE"

        # output to the cv2
        return_res.append([top, right, bottom, left, gender_text, age_text])

    return return_res


class FaceDetect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        frame = self.vs.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = detect_face(rgb_frame)

        # Display the results
        for (
            top,
            right,
            bottom,
            left,
            gender_text,
            age_text,
        ) in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Sex: {}".format(gender_text),
                (left + 25, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            cv2.putText(
                frame,
                "Age: {}".format(age_text),
                (left + 25, top - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        ret, jpeg = cv2.imencode(".jpg", frame)

        return jpeg.tobytes()

    def get_text(self):
        frame = self.vs.read()
        return_res = []

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = detect_face(rgb_frame)

        # Display the results
        for (
            top,
            right,
            bottom,
            left,
            gender_text,
            age_text,
        ) in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Sex: {}".format(gender_text),
                (left + 25, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            cv2.putText(
                frame,
                "Age: {}".format(age_text),
                (left + 25, top - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        return_res.append([gender_text, age_text])

        return return_res
