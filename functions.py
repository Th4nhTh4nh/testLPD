import os
import cv2
import numpy as np
import util
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

# định nghĩa các đường dẫn đến cfg, weights , nameclass của model
MODEL_CFG_PATH = os.path.join(".", "model", "cfg", "yolov3-tiny.cfg")
MODEL_WEIGHTS_PATH = os.path.join(".", "model", "weights", "yolov3-tiny_15000.weights")
CLASS_NAME_PATH = os.path.join(".", "model", "class.names")
# folder lưu ảnh từng ký tự
SAVE_DIR = "./char_imgs"
# đường dẫn ảnh
IMAGE_PATH = "./pic/car1.jpg"
# Đường dẫn đến mô hình CNN
MODEL_PATH = "./model/cnn/model2-200.keras"
# Đường dẫn đến thư mục chứa ảnh cần dự đoán
IMAGE_FOLDER = "./char_imgs"
# lớp của ký tự
class_ky_tu = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# phát hiện và cắt ảnh biển số
def DetectPlate(image):
    bboxes = []
    class_ids = []
    scores = []

    net = cv2.dnn.readNetFromDarknet(MODEL_CFG_PATH, MODEL_WEIGHTS_PATH)

    img = cv2.imread(image)
    H, W, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    net.setInput(blob)
    detections = util.get_outputs(net)
    for detection in detections:
        bbox = detection[:4]
        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]
        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        # lấy tọa độ của biển số xe
        license_plate = img[
            int(yc - (h / 2)) : int(yc + (h / 2)),
            int(xc - (w / 2)) : int(xc + (w / 2)),
            :,
        ].copy()
        img = cv2.rectangle(
            img,
            (int(xc - (w / 2)), int(yc - (h / 2))),
            (int(xc + (w / 2)), int(yc + (h / 2))),
            (0, 255, 0),
            10,
        )
        # xử lý ảnh biển số xe trước khi nhận diện kí tự
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(
            license_plate_gray, 103, 255, cv2.THRESH_BINARY_INV
        )
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # license_plate_thresh = cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB)

    #plt.figure()
    #plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
    #plt.show()
    return license_plate_gray, license_plate_thresh


# tiền xử lý, lọc nhiễu trên ảnh biển số
def PlatePreprecess(license_plate_thresh):
    _, labels = cv2.connectedComponents(license_plate_thresh)
    mask = np.zeros(license_plate_thresh.shape, dtype="uint8")
    total_pixels = license_plate_thresh.shape[0] * license_plate_thresh.shape[1]
    lower = total_pixels // 120
    upper = total_pixels // 20

    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(license_plate_thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)
    plt.figure()
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.show()
    return mask


# phân tách ký tự
def CharacterSegment(license_plate_thresh):
    count = 0
    # Tách từng ký tự từ vùng chứa ảnh của biển số xe
    contours, _ = cv2.findContours(
        license_plate_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
    boundingBoxes = np.array(boundingBoxes)

    mean_w = np.mean(boundingBoxes[:, 2])
    mean_h = np.mean(boundingBoxes[:, 3])
    mean_y = np.mean(boundingBoxes[:, 1])

    threshold_w = mean_w * 1.5
    threshold_h = mean_h * 1.5

    new_boundingBoxes = boundingBoxes[(boundingBoxes[:, 2] < threshold_w) & (boundingBoxes[:, 3] < threshold_h)]

    #image_with_bbox = cv2.cvtColor(license_plate_thresh.copy(), cv2.COLOR_GRAY2BGR)
    """"
    for bbox in new_boundingBoxes:
        x, y, w, h = bbox
        image_with_bbox = cv2.rectangle(image_with_bbox, (x, y), (x+w, y+h), (0, 0, 255), 2)
        image_with_bbox_on_black = np.zeros_like(image_with_bbox)
        image_with_bbox_on_black = cv2.rectangle(image_with_bbox_on_black, (0, 0),
                                                 (image_with_bbox.shape[1], image_with_bbox.shape[0]), (0, 0, 0), -1)
        image_with_bbox_on_black = cv2.add(image_with_bbox_on_black, image_with_bbox)

        cv2.imshow("Image with Bounding Box", image_with_bbox_on_black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    new_boundingBoxes = sorted(new_boundingBoxes, key=lambda bbox: bbox[0])

    for i, bbox in enumerate(new_boundingBoxes):
        x, y, w, h = bbox
        char_img = license_plate_thresh[y:y+h, x:x+w]
        char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)

        filename = f"char_{i}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, char_img)

    #image_with_bbox = cv2.cvtColor(license_plate_thresh.copy(), cv2.COLOR_GRAY2BGR)

    # for bbox in boundingBoxes:
    #     x, y, w, h = bbox
    #     image_with_bbox = cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Hiển thị ảnh đã vẽ bounding box trên nền đen
    # image_with_bbox_on_black = np.zeros_like(image_with_bbox)
    # image_with_bbox_on_black = cv2.rectangle(image_with_bbox_on_black, (0, 0),
    #                                          (image_with_bbox.shape[1], image_with_bbox.shape[0]), (0, 0, 0), -1)
    # image_with_bbox_on_black = cv2.add(image_with_bbox_on_black, image_with_bbox)
    #
    # cv2.imshow("Image with Bounding Box", image_with_bbox_on_black)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # filtered_contours = [
    #     contour
    #     for contour in contours
    #     if (cv2.contourArea(contour) > 30 and cv2.contourArea(contour) < 400)
    # ]
    # filtered_contours.sort(key=lambda x: cv2.boundingRect(x)[0])
    #
    # for contour in filtered_contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     #character = license_plate_thresh[y : y + h, x : x + w]
    #     character = cv2.resize(license_plate_thresh[y: y + h, x: x + w], (20, 20))
    #     count += 1
    #     filename = f"char_{count}.jpg"
    #     save_path = os.path.join(SAVE_DIR, filename)
    #     cv2.imwrite(save_path, character)
    #
    #     # Hiển thị từng ký tự
    #     plt.figure()
    #     plt.imshow(character, cmap="gray")
    #     plt.title(f"Character {len(filtered_contours)}")
    #     plt.show()


# thực hiện phấn lớp ký tự
def CharactersClassification():
    model = load_model(MODEL_PATH)
    Bien_so_xe = []
    # Duyệt qua từng ảnh trong thư mục
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith(".jpg"):
            char_img_path = os.path.join(IMAGE_FOLDER, filename)

            img = image.load_img(char_img_path, target_size=(20, 20))
            img_array = image.img_to_array(img)
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            gray_img = gray_img.reshape((20, 20, 1))

            prediction = model.predict(np.array([gray_img]))
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_ky_tu[predicted_class_index]
            Bien_so_xe.append(class_ky_tu[predicted_class_index])
            confidence = np.max(prediction)
            print(
                f"File: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence}"
            )

    result_string = "".join(Bien_so_xe)
    print("biển số xe: ", result_string)
    return result_string


detect, plate = DetectPlate(IMAGE_PATH)
processed_plate = PlatePreprecess(plate)
CharacterSegment(processed_plate)
CharactersClassification()
