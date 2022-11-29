import boto3
import requests
import cv2
import numpy as np
import os
from typing import Union, AnyStr, List
from requests_toolbelt.multipart import MultipartEncoder
import io
from PIL import Image
from datetime import datetime
import glob


def roboflow(frame: Union[AnyStr, List[AnyStr]]) -> tuple:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(image)
    LT_x, LT_y, RB_x, RB_y = 0, 0, 0, 0

    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")
    multi_encoder = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
    response = requests.post("https://detect.roboflow.com/lic-plates/2?api_key=DmdaAV7Z3yTdZAk5GiNI",
                             data=multi_encoder, headers={'Content-Type': multi_encoder.content_type})
    resp = response.json()
    length = len(resp.get('predictions'))
    bbox, pred_conf = list(), 0

    for i in range(length):
        pred_x = resp.get('predictions')[i]['x']
        pred_y = resp.get('predictions')[i]['y']
        pred_w = resp.get('predictions')[i]['width']
        pred_h = resp.get('predictions')[i]['height']
        pred_conf = resp.get('predictions')[i]['confidence']
        bbox.append((pred_x, pred_y, pred_w, pred_h, pred_conf))

        for item in bbox:
            x, y, w, h, conf = item
            LT_x = np.subtract(x, np.floor_divide(w, 2)).astype('int')
            LT_y = np.subtract(y, np.floor_divide(h, 2)).astype('int')
            RB_x = np.add(x, np.floor_divide(w, 2)).astype('int')
            RB_y = np.add(y, np.floor_divide(h, 2)).astype('int')

    return LT_x, LT_y, RB_x, RB_y, pred_conf


def extract_image_from_text(filename: Union[AnyStr, List[AnyStr]]):
    client = boto3.client('textract', region_name='us-east-2')
    with open(filename, mode='rb') as imgfile:
        imgBytes = bytearray(imgfile.read())

    result = client.detect_document_text(Document={'Bytes': imgBytes})
    elements, text_arr = result['Blocks'], list()

    for item in elements:
        if item["BlockType"] == "LINE":
            text_arr.append(item['Text'])

    for text in text_arr:
        return text


test_image = cv2.imread(os.path.join('IMAGES', 'random', np.random.choice(glob.glob(r'C:\Users\ShrikantViswanathan\Documents\Random_LPR_Test_Images\*'))))
test_image = cv2.resize(test_image, (700, 600))
x, y, w, h, conf = roboflow(test_image)

current_time = datetime.strptime(str(datetime.now()).split(".")[0], '%Y-%m-%d %H:%M:%S')
current_output_path: Union[AnyStr, List[AnyStr]] =\
    f'test_output({current_time.hour}-{current_time.minute}-{current_time.second}).png'
print(f'confidence level: {np.multiply(conf, 100).astype("int")}'.capitalize())

if np.greater_equal(conf, 0.6):
    cv2.rectangle(test_image, (x + 10, y), (w, h), (255, 0, 0), 3, cv2.LINE_8)
    cv2.imwrite(os.path.join('IMAGES', 'random', 'outputs', current_output_path), test_image[y: h, x: w])
    print('saved image')

image_reader = extract_image_from_text(os.path.join("IMAGES", "random", "outputs", current_output_path))
print(f'Text: {image_reader}')
cv2.putText(test_image, str(image_reader) if image_reader is not None else "None",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow(f'test for {current_output_path.split("_")[1]}'.capitalize(), test_image)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyWindow('test')
