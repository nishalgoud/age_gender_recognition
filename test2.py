import cv2
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Defined the model files
FACE_PROTO = "weights/opencv_face_detector.pbtxt"
FACE_MODEL = "weights/opencv_face_detector_uint8.pb"

AGE_PROTO = "weights/age_deploy.prototxt"
AGE_MODEL = "weights/age_net.caffemodel"

GENDER_PROTO = "weights/gender_deploy.prototxt"
GENDER_MODEL = "weights/gender_net.caffemodel"

# Load network
FACE_NET = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
AGE_NET = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
GENDER_NET = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

box_padding = 20

def get_face_box (net, frame, conf_threshold = 0.7):
    frame_copy = frame.copy()
    frame_height = frame_copy.shape[0]
    frame_width = frame_copy.shape[1]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

    return frame_copy, boxes
def age_gender_detector(input_path):
    image = cv2.imread(input_path)
    resized_image = cv2.resize(image, (640, 480))

    frame = resized_image.copy()
    frame_face, boxes = get_face_box(FACE_NET, frame)

    age_predictions_list = []
    gender_predictions_list = []

    for box in boxes:
        face = frame[max(0, box[1] - box_padding):min(box[3] + box_padding, frame.shape[0] - 1), \
              max(0, box[0] - box_padding):min(box[2] + box_padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        GENDER_NET.setInput(blob)
        gender_predictions = GENDER_NET.forward()
        gender = GENDER_LIST[gender_predictions[0].argmax()]

        AGE_NET.setInput(blob)
        age_predictions = AGE_NET.forward()
        age = AGE_LIST[age_predictions[0].argmax()]

        age_predictions_list.append(age)
        gender_predictions_list.append(gender)

    return age_predictions_list, gender_predictions_list

if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = "output/"

    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)]
    age_predictions_list = []
    gender_predictions_list = []

    for image_path in image_paths:
        age_preds, gender_preds = age_gender_detector(image_path)
        age_predictions_list.extend(age_preds)
        gender_predictions_list.extend(gender_preds)
        #input_path = os.path.join(input_folder, filename)
        #output = age_gender_detector(input_path, age_predictions_list, gender_predictions_list)
        #output_path = os.path.join(output_folder, filename)
        #cv2.imwrite(output_path, output)

        

    # Convert the lists to pandas Series objects
    age_predictions_series = pd.Series(age_predictions_list)
    gender_predictions_series = pd.Series(gender_predictions_list)

    # Plotting age and gender distributions
    plt.figure()
    #sns.countplot(age_predictions_series, order=AGE_LIST)
    sns.countplot(x=age_predictions_series, order=AGE_LIST)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_folder, "age_distribution.png"))

    plt.figure()
    #sns.countplot(gender_predictions_series, order=GENDER_LIST)
    sns.countplot(x=gender_predictions_series, order=GENDER_LIST)
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_folder, "gender_distribution.png"))

    plt.show()