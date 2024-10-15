import string
import easyocr
import logging
from rdw.rdw import Rdw
from matplotlib import pyplot as plt
import os


reader = easyocr.Reader(['nl'], gpu=False)
logging.basicConfig(filename='kentekens.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def license_complies_format(text):
    # Remove all special characters and spaces
    text = text.replace('-', '')
    licence_plate_clean = ''.join(filter(lambda x: x.isalnum() and x in string.ascii_letters + string.digits, text))
    
    if len(licence_plate_clean) != 6:
        return False, None, None

    try:
        # Check if the license plate exists in the RDW database
        car = Rdw()
        car_exist = car.get_vehicle_data(licence_plate_clean)
        
        if len(car_exist) != 0:
            car_values = [car_exist[0]['voertuigsoort'], car_exist[0]['merk']]
            return True, licence_plate_clean, car_values
    
    except Exception as e:
        logging.error(f"Error: {e}")
    
    return False, None, None


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        result = license_complies_format(text)
        
        if result[0]:
            logging.info(f"Gedetecteerde tekst: {result[1]}, Score: {score}")
            return result[1], score, result[2]
        
    return None, None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def load_last_frame():
    frame_file = '/Users/vincent/Downloads/IFFDA bak/Automatic-License-Plate-Recognition-using-YOLOv8/frame_checkpoint.txt'
    if os.path.exists(frame_file):
        with open(frame_file, 'r') as f:
            frame_num = f.read().strip()
            if frame_num.isdigit():
                return int(frame_num)
    return 0  # Begin vanaf het eerste frame als er geen bestand is

def save_frame_number(frame_num):
    frame_file = '/Users/vincent/Downloads/IFFDA bak/Automatic-License-Plate-Recognition-using-YOLOv8/frame_checkpoint.txt'
    with open(frame_file, 'w') as f:
        f.write(str(frame_num))