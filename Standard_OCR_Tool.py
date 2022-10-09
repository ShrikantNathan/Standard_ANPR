import cv2
import numpy as np
import os, io
from typing import List, Tuple, Union
import easyocr, pytesseract
import requests
from PIL import Image
from requests_toolbelt import MultipartEncoder

# print(subprocess.call("py -V"))
# print(subprocess.call("conda -V"))
blank_screen = np.zeros((768, 1366, 3), dtype=np.uint8)
blank_screen_2 = np.ones((768, 1366), dtype=np.uint8)
blank_screen_3 = np.zeros((768, 1366), dtype=np.uint8)
blank_screen_4 = np.zeros((768, 1366), dtype=np.uint8)

screen_1 = cv2.imread(f"{os.path.join(os.getcwd(), 'randompics')}"
                      f"/{np.random.choice([file for file in os.listdir(f'{os.getcwd()}/randompics')][:3])}")
screen_2 = cv2.imread(f"{os.path.join(os.getcwd(), 'randompics')}"
                      f"/{np.random.choice([file for file in os.listdir(f'{os.getcwd()}/randompics')][3:6])}")
screen_3 = cv2.imread(f"{os.path.join(os.getcwd(), 'randompics')}"
                      f"/{np.random.choice([file for file in os.listdir(f'{os.getcwd()}/randompics')][6:9])}")
screen_4 = cv2.imread(f"{os.path.join(os.getcwd(), 'randompics')}"
                      f"/{np.random.choice([file for file in os.listdir(f'{os.getcwd()}/randompics')])}")


def set_constant_shapes(orig_image) -> Tuple[int, float]:   # not in use
    return np.floor_divide(orig_image.shape[1], 2).astype('int'),\
           np.floor_divide(orig_image.shape[0], 2).astype('int')


SCREEN_HEIGHT, SCREEN_WIDTH = 768, 1366
scaled_screen_1 = cv2.resize(screen_1, dsize=(int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2)))
scaled_screen_2 = cv2.resize(screen_2, dsize=(int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2)))
scaled_screen_3 = cv2.resize(screen_3, dsize=(int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2)))
scaled_screen_4 = cv2.resize(screen_4, dsize=(int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2)))

scaled_screen_1_gray = cv2.cvtColor(scaled_screen_1, cv2.COLOR_BGR2GRAY)
scaled_screen_2_gray = cv2.cvtColor(scaled_screen_2, cv2.COLOR_BGR2GRAY)
scaled_screen_3_gray = cv2.cvtColor(scaled_screen_3, cv2.COLOR_BGR2GRAY)
scaled_screen_4_gray = cv2.cvtColor(scaled_screen_4, cv2.COLOR_BGR2GRAY)

scaled_screen_1_blurred = cv2.medianBlur(scaled_screen_1_gray, ksize=5)
scaled_screen_2_blurred = cv2.medianBlur(scaled_screen_2_gray, ksize=5)
scaled_screen_3_blurred = cv2.medianBlur(scaled_screen_3_gray, ksize=5)
scaled_screen_4_blurred = cv2.medianBlur(scaled_screen_4_gray, ksize=5)

scaled_screen_1_edged = cv2.Canny(scaled_screen_1_blurred, threshold1=10, threshold2=200)
scaled_screen_2_edged = cv2.Canny(scaled_screen_2_blurred, threshold1=10, threshold2=200)
scaled_screen_3_edged = cv2.Canny(scaled_screen_3_blurred, threshold1=10, threshold2=200)
scaled_screen_4_edged = cv2.Canny(scaled_screen_4_blurred, threshold1=10, threshold2=200)

contours, ret = cv2.findContours(scaled_screen_1_edged, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
contours2, ret2 = cv2.findContours(scaled_screen_2_edged, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
contours3, ret3 = cv2.findContours(scaled_screen_3_edged, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
contours4, ret4 = cv2.findContours(scaled_screen_4_edged, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:5]
contours3 = sorted(contours3, key=cv2.contourArea, reverse=True)[:5]
contours4 = sorted(contours4, key=cv2.contourArea, reverse=True)[:5]


def approx_and_predict_contour(test_contours: Union[tuple, List[tuple]]) -> Union[int, List[int]]:
    # loop over the contours
    n_plate_cnt = 0
    for c in test_contours:
        peri = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(curve=c, epsilon=0.02 * peri, closed=True)
        # if the contour has 4 points, then we can say that its a license plate
        if np.equal(len(approx), 4):
            n_plate_cnt = approx
            break
        else:
            continue
    return n_plate_cnt


def Roboflow(frame):
    ''' License Plate '''
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(image)
    LT_x, LT_y, RB_x, RB_y = 0, 0, 0, 0

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")

    # Build multipart form and post request
    m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

    response = requests.post("https://detect.roboflow.com/lic-plates/2?api_key=DmdaAV7Z3yTdZAk5GiNI", data=m,
                             headers={'Content-Type': m.content_type})
    resp = response.json()

    length = len(resp.get('predictions'))
    pred_conf, bbox = 0, []

    for i in range(length):
        # this is for center point coordinates.
        pred_x = resp.get('predictions')[i]['x']
        pred_y = resp.get('predictions')[i]['y']
        pred_w = resp.get('predictions')[i]['width']
        pred_h = resp.get('predictions')[i]['height']
        pred_conf = resp.get('predictions')[i]['confidence']
        bbox.append((pred_x, pred_y, pred_w, pred_h, pred_conf))

        for item in bbox:
            x, y, w, h, conf = item

            LT_x = int(x - (w / 2))
            LT_y = int(y - (h / 2))

            RB_x = int(x + (w / 2))
            RB_y = int(y + (h / 2))

    return LT_x, LT_y, RB_x, RB_y, pred_conf


x, y, w, h, conf = Roboflow(scaled_screen_1_gray)
x2, y2, w2, h2, conf2 = Roboflow(scaled_screen_2_gray)
x3, y3, w3, h3, conf3 = Roboflow(scaled_screen_3_gray)
x4, y4, h4, w4, conf4 = Roboflow(scaled_screen_4_gray)

lic_plate_portion1 = scaled_screen_1_gray[y: y + h, x: x + w]
lic_plate_portion2 = scaled_screen_2_gray[y2: y2 + h2, x2: x2 + w2]
lic_plate_portion3 = scaled_screen_3_gray[y3: y3 + h3, x3: x3 + w3]
lic_plate_portion4 = scaled_screen_4_gray[y4: y4 + h4, x4: x4 + w4]

cv2.rectangle(scaled_screen_1, (x, y), (x - w//2, y + h//2), (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.rectangle(scaled_screen_2, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.rectangle(scaled_screen_3, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.rectangle(scaled_screen_4, (x4, y4), (x4 + w4, y4 + h4), (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

if not os.path.exists("Cropped_License_Plates"):
    os.makedirs('Cropped_License_Plates')

cv2.imwrite(f"{os.getcwd()}/Cropped_License_Plates/Detected-1.jpg", lic_plate_portion1)
cv2.imwrite(f"{os.getcwd()}/Cropped_License_Plates/Detected-2.jpg", lic_plate_portion2)
cv2.imwrite(f"{os.getcwd()}/Cropped_License_Plates/Detected-3.jpg", lic_plate_portion3)
cv2.imwrite(f"{os.getcwd()}/Cropped_License_Plates/Detected-4.jpg", lic_plate_portion4)
print("Proposed License plate ROI locations saved..")

reader = easyocr.easyocr.Reader(lang_list=["en"])
# detect the text from the license plates
# detection, detection2 = reader.readtext(lic_plate_portion1), reader.readtext(lic_plate_portion2)
# detection3, detection4 = reader.readtext(lic_plate_portion3), reader.readtext(lic_plate_portion4)
detection, detection2 = pytesseract.image_to_string(lic_plate_portion1), pytesseract.image_to_string(lic_plate_portion2)
detection3, detection4 = pytesseract.image_to_string(lic_plate_portion3), pytesseract.image_to_string(lic_plate_portion4)

if detection is not None:
    cv2.putText(scaled_screen_1, str(detection), (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), thickness=3)
if detection2 is not None:
    cv2.putText(scaled_screen_2, str(detection2), (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), thickness=3)
if detection3 is not None:
    cv2.putText(scaled_screen_3, str(detection3), (x3 - 10, y3 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), thickness=3)
if detection4 is not None:
    cv2.putText(scaled_screen_4, str(detection4), (x4 - 10, y4 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), thickness=3)

gray_stack_1 = np.hstack((scaled_screen_1_gray, scaled_screen_2_gray))
gray_stack_2 = np.hstack((scaled_screen_3_gray, scaled_screen_4_gray))
blur_stack_1 = np.hstack((scaled_screen_1_blurred, scaled_screen_2_blurred))
blur_stack_2 = np.hstack((scaled_screen_3_blurred, scaled_screen_4_blurred))
edge_stack_1 = np.hstack((scaled_screen_1_edged, scaled_screen_2_edged))
edge_stack_2 = np.hstack((scaled_screen_3_edged, scaled_screen_4_edged))

# Now allocating different portions of blank screens for these scaled screens
blank_screen[0: blank_screen.shape[0] // 2, 0: blank_screen.shape[1] // 2] = scaled_screen_1
blank_screen[0: blank_screen.shape[0] // 2, blank_screen.shape[1] // 2: blank_screen.shape[1]] = scaled_screen_2
blank_screen[blank_screen.shape[0] // 2: blank_screen.shape[0], blank_screen.shape[1] // 2: blank_screen.shape[1]] = scaled_screen_3
blank_screen[blank_screen.shape[0] // 2: blank_screen.shape[0], 0: blank_screen.shape[1] // 2] = scaled_screen_4

blank_screen_2[0: np.floor_divide(blank_screen_2.shape[0], 2), 0: blank_screen_2.shape[1]] = gray_stack_1
blank_screen_2[blank_screen_2.shape[0] // 2: blank_screen_2.shape[0], 0: blank_screen_2.shape[1]] = gray_stack_2
blank_screen_3[0: np.floor_divide(blank_screen_3.shape[0], 2), 0: blank_screen_3.shape[1]] = blur_stack_1
blank_screen_3[np.floor_divide(blank_screen_3.shape[0], 2): blank_screen_3.shape[0], 0: blank_screen_3.shape[1]] = blur_stack_2
blank_screen_4[0: np.floor_divide(blank_screen_4.shape[0], 2), 0: blank_screen_4.shape[1]] = edge_stack_1
blank_screen_4[np.floor_divide(blank_screen_4.shape[0], 2): blank_screen_4.shape[0], 0: blank_screen_4.shape[1]] = edge_stack_2


def raw_plan_layout():
    # Screen 1 allocation
    cv2.circle(blank_screen, center=(0, 0), radius=20, color=(255, 0, 0), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, 0), radius=20, color=(255, 0, 0), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(0, blank_screen.shape[0] // 2), radius=20, color=(255, 0, 0), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, blank_screen.shape[0] // 2), radius=20, color=(255, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(blank_screen, (0, 0), (blank_screen.shape[1] // 2, blank_screen.shape[0] // 2), color=(255, 0, 0), thickness=3)

    # Screen 2 allocation
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, 0), radius=17, color=(0, 255, 0), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1], 0), radius=17, color=(0, 255, 0), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, blank_screen.shape[0] // 2), radius=17, color=(0, 255, 0), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1], blank_screen.shape[0] // 2), radius=17, color=(0, 255, 0), thickness=cv2.FILLED)
    cv2.rectangle(blank_screen, (blank_screen.shape[1] // 2, 0), (blank_screen.shape[1], blank_screen.shape[0] // 2), color=(0, 255, 0), thickness=3)

    # Screen 3 allocation
    cv2.circle(blank_screen, center=(0, blank_screen.shape[0] // 2), radius=14, color=(0, 255, 255), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, blank_screen.shape[0] // 2), radius=14, color=(0, 255, 255), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(0, blank_screen.shape[0]), radius=14, color=(0, 255, 255), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, blank_screen.shape[0]), radius=14, color=(0, 255, 255), thickness=cv2.FILLED)
    cv2.rectangle(blank_screen, (0, blank_screen.shape[0] // 2), (blank_screen.shape[1] // 2, blank_screen.shape[0]), color=(0, 255, 255), thickness=3)

    # Screen 4 allocation
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, blank_screen.shape[0] // 2), radius=14, color=(0, 0, 255), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1], blank_screen.shape[0] // 2), radius=14, color=(0, 0, 255), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 2, blank_screen.shape[0]), radius=14, color=(0, 0, 255), thickness=cv2.FILLED)
    cv2.circle(blank_screen, center=(blank_screen.shape[1], blank_screen.shape[0]), radius=14, color=(0, 0, 255), thickness=cv2.FILLED)
    cv2.rectangle(blank_screen, (blank_screen.shape[1] // 2, blank_screen.shape[0] // 2), (blank_screen.shape[1], blank_screen.shape[0]), color=(0, 0, 255), thickness=3)

    # line numbering
    cv2.putText(blank_screen, text="1", fontScale=3.5, fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, org=((blank_screen.shape[1] // 4) - 30, (blank_screen.shape[0] // 4) + 20), color=(255, 0, 0), thickness=5)
    cv2.putText(blank_screen, text="2", fontScale=3.5, fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, org=(3 * (blank_screen.shape[1] // 4) - 35, (blank_screen.shape[0] // 4) + 20), color=(0, 255, 0), thickness=5)
    cv2.putText(blank_screen, text="3", fontScale=3.5, fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, org=((blank_screen.shape[1] // 4) - 35, 3 * (blank_screen.shape[0] // 4) + 20), color=(0, 255, 255), thickness=5)
    cv2.putText(blank_screen, text="4", fontScale=3.5, fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX, org=(3 * (blank_screen.shape[1] // 4) - 35, 3 * (blank_screen.shape[0] // 4) + 20), color=(0, 0, 255), thickness=5)

    # Line encircling
    cv2.circle(blank_screen, center=(blank_screen.shape[1] // 4, (blank_screen.shape[0] // 4)-10), radius=75, thickness=4, color=(255, 0, 0))
    cv2.circle(blank_screen, center=(3 * (blank_screen.shape[1] // 4), (blank_screen.shape[0] // 4)-10), radius=75, thickness=4, color=(0, 255, 0))
    cv2.circle(blank_screen, center=((blank_screen.shape[1] // 4), 3 * (blank_screen.shape[0] // 4)-10), radius=75, thickness=4, color=(0, 255, 255))
    cv2.circle(blank_screen, center=(3 * (blank_screen.shape[1] // 4), 3 * (blank_screen.shape[0] // 4)-10), radius=75, thickness=4, color=(0, 0, 255))

    cv2.imshow('raw layout plan', blank_screen)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        print('exiting the current window...')
        cv2.destroyAllWindows()
        print('screen closed'.capitalize())


cv2.imshow('blank', blank_screen)
# cv2.imshow('Grayscale', blank_screen_2)
# cv2.imshow('Blurred', blank_screen_3)
# cv2.imshow("Edged", blank_screen_4)
# raw_plan_layout()
cv2.waitKey(0)