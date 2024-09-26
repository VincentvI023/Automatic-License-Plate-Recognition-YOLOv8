import string
import easyocr
import logging
from rdw.rdw import Rdw

from matplotlib import pyplot as plt

# Initialize the OCR reader
reader = easyocr.Reader(['nl'], gpu=False)

logging.basicConfig(filename='kentekens.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'Q': '0'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '5': 'S',
                    '4': 'A',
                    '6': 'G',
                    '2': 'Z'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                # print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    text = text.replace('-', '')
    licence_plate_clean = ''.join(filter(lambda x: x.isalnum() and x in string.ascii_letters + string.digits, text))
    
    if len(licence_plate_clean) != 6:
        return False, licence_plate_clean
    
    # Check if the license plate text complies with Dutch license plate formats
    # Dutch license plates can have several formats, such as:
    # - XX-99-99
    # - 99-99-XX
    # - 99-XX-99
    # - XX-99-XX
    # - XX-XX-99
    # - 99-XX-XX
    # Where X is a letter and 9 is a digit

    # formats = [
    #     (string.ascii_uppercase, string.ascii_uppercase, string.digits, string.digits, string.digits, string.digits),
    #     (string.digits, string.digits, string.digits, string.digits, string.ascii_uppercase, string.ascii_uppercase),
    #     (string.digits, string.digits, string.ascii_uppercase, string.ascii_uppercase, string.digits, string.digits),
    #     (string.ascii_uppercase, string.ascii_uppercase, string.digits, string.digits, string.ascii_uppercase, string.ascii_uppercase),
    #     (string.ascii_uppercase, string.ascii_uppercase, string.ascii_uppercase, string.ascii_uppercase, string.digits, string.digits),
    #     (string.digits, string.digits, string.ascii_uppercase, string.ascii_uppercase, string.ascii_uppercase, string.ascii_uppercase)
    # ]
    
    # for fmt in formats:
    #     if all(text[i] in fmt[i] for i in range(6)):
    #         return True
    # else:
    #     return False
    car = Rdw()
    car_exist = car.get_vehicle_data(licence_plate_clean)
    
    if len(car_exist) != 0:
        print(car_exist[0]['voertuigsoort'])
        print(car_exist[0]['merk'])
        
        if car_exist[0]['tweede_kleur'] != "Niet geregistreerd":
            print(car_exist[0]['tweede_kleur'])
        else:
            print(car_exist[0]['eerste_kleur'])
        return True, licence_plate_clean
    
    return False, licence_plate_clean


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return text


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        result = license_complies_format(text)
        
        if result[0]:
            logging.info(f"Gedetecteerde tekst: {result[1]}, Score: {score}")
            print(f"Gedetecteerde tekst: {result[1]}, Score: {score}")
            return result[1], score
        
    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
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
