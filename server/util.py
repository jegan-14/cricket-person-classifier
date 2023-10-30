import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import warnings

warnings.filterwarnings("ignore")

__class_number_to_name = {}
__class_name_to_number = {}
__model = None


def classify_image(base64, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, base64)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        harr_image = w2d(img, "db1", 5)
        scalled_harr_img = cv2.resize(harr_image, (32, 32))

        combined = np.vstack(
            (
                scalled_raw_img.reshape(32 * 32 * 3, 1),
                scalled_harr_img.reshape(32 * 32, 1),
            )
        )
        length_of_img = 32 * 32 * 3 + 32 * 32

        final_img = combined.reshape(1, length_of_img).astype(float)

        result.append(
            {
                "class": class_name_from_number(__model.predict(final_img)[0]),
                "class_probability": np.round(
                    __model.predict_proba(final_img) * 100
                ).tolist()[0],
                "class_dictinary": __class_name_to_number,
            }
        )
    return result


def class_name_from_number(class_number):
    return __class_number_to_name[class_number]


def load_artifacts():
    print("Loading the artifacts...")

    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class labels.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {j: i for i, j in __class_name_to_number.items()}

    global __model

    with open("./artifacts/model.pkl", "rb") as f:
        __model = joblib.load(f)

    print("Finished loading the artifacts")


def get_cv2_image_from_base64(b64str):
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path, base64_data):
    face_cascade = cv2.CascadeClassifier(
        "./haarcascade/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)

    else:
        img = get_cv2_image_from_base64(base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    cropped_faces = []
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def convert_image_base64_test():
    with open("base64.txt") as f:
        return f.read()


if __name__ == "__main__":
    load_artifacts()
    # print(classify_image(convert_image_base64_test(), None))
    # print(classify_image(None, "./test images/ben1.jpg"))
    # print(classify_image(None, "./test images/dhoni1.jpg"))
    # print(classify_image(None, "./test images/sachin1.jpg"))
    # print(classify_image(None, "./test images/virat1.jpg"))
    # print(classify_image(None, "./test images/smith1.jpg"))
    print(classify_image(None, "./test images/sachin2.jpg"))
