from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import os
import csv


plate_folder = '/plates'
results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./real-sample.mov')

# Clear all rows except the first one in the CSV file
csv_file_path = './test.csv'
with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)

with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        if len(detections_) > 0: 
            track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                cv2.imwrite(os.path.join(plate_folder, f'frame_{frame_nmr}_car_{car_id}_crop.jpg'), license_plate_crop)

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(plate_folder, f'frame_{frame_nmr}_car_{car_id}_gray.jpg'), license_plate_crop_gray)

                license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
                cv2.imwrite(os.path.join(plate_folder, f'frame_{frame_nmr}_car_{car_id}_thresh.jpg'), license_plate_crop_thresh)

                # Additional processing variants
                license_plate_crop_blur = cv2.GaussianBlur(license_plate_crop_gray, (5, 5), 0)
                cv2.imwrite(os.path.join(plate_folder, f'frame_{frame_nmr}_car_{car_id}_blur.jpg'), license_plate_crop_blur)

                _, license_plate_crop_binary = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(plate_folder, f'frame_{frame_nmr}_car_{car_id}_binary.jpg'), license_plate_crop_binary)

                license_plate_crop_median = cv2.medianBlur(license_plate_crop_gray, 5)
                cv2.imwrite(os.path.join(plate_folder, f'frame_{frame_nmr}_car_{car_id}_median.jpg'), license_plate_crop_median)

                kernel = np.ones((3, 3), np.uint8)
                license_plate_crop_morph = cv2.morphologyEx(license_plate_crop_thresh, cv2.MORPH_CLOSE, kernel)
                cv2.imwrite(os.path.join(plate_folder, f'frame_{frame_nmr}_car_{car_id}_morph.jpg'), license_plate_crop_morph)

                # Perform OCR on different variants of the license plate
                license_plate_text_variants = [
                    read_license_plate(license_plate_crop),
                    read_license_plate(license_plate_crop_gray),
                    read_license_plate(license_plate_crop_thresh),
                    read_license_plate(license_plate_crop_blur),
                    read_license_plate(license_plate_crop_binary),
                    read_license_plate(license_plate_crop_median),
                    read_license_plate(license_plate_crop_morph)
                ]

                # Select the best variant based on the text score
                license_plate_text, license_plate_text_score = max(license_plate_text_variants, key=lambda x: x[1] if x[0] is not None else 0)
                print(license_plate_text, license_plate_text_score)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    # Write results to CSV
                    with open(csv_file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([frame_nmr, car_id, xcar1, ycar1, xcar2, ycar2, x1, y1, x2, y2, license_plate_text, score, license_plate_text_score])