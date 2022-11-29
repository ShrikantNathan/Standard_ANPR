import os
import cv2
import numpy as np
import boto3
from PIL import Image
import io
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
from append_json import append_all_contents_2
from store_json_event import store_detected_events_in_corr_json
from datetime import datetime, timedelta
import random
from typing import Union, List, AnyStr


def process_json_with_correspondent_scene(scene: str, scene_dir: Union[AnyStr, List[AnyStr]],
                                          json_view_dir: Union[AnyStr, List[AnyStr]]) -> Union[str, List[str]]:
    json_renderer = str()
    for i in range(1, len(scene_dir)):
        for j in range(1, len(json_view_dir)):
            if scene == f'Scene_{j}':
                json_renderer = f'Scene_{j}.json'
                break
        break
    print(f'json_file selected as: {json_renderer}.')
    return json_renderer


class AutomaticNumberPlateRecognition:
    def __init__(self):
        self.client = boto3.client('textract', region_name="us-east-2")
        self.screen_width: Union[int, float] = 1920
        self.screen_height: Union[int, float] = 1080
        self.scene = random.choice(['Scene_3', 'Scene_4', 'Scene_6', 'Scene_7', 'Scene_8'])
        self.corr_json_files = list(file for file in os.listdir("scenes_json"))
        self.total_scenes = list(scene for scene in os.listdir(os.path.join('Scene', 'BusStopArm_Video')))[1:]
        self.json_file = process_json_with_correspondent_scene(self.scene, self.total_scenes, self.corr_json_files)
        self.video2 = rf'{os.path.join(os.getcwd(), "Scene", "BusStopArm_Video")}\{self.scene}\Cam_2.mp4'
        self.video3 = rf'{os.path.join(os.getcwd(), "Scene", "BusStopArm_Video")}\{self.scene}\Cam_3.mp4'
        self.video4 = rf'{os.path.join(os.getcwd(), "Scene", "BusStopArm_Video")}\{self.scene}\Cam_4.mp4'
        self.video2_cap = cv2.VideoCapture(self.video2)
        self.video3_cap = cv2.VideoCapture(self.video3)
        self.video4_cap = cv2.VideoCapture(self.video4)
        self.blank = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.output_path: Union[str, List[AnyStr]] = str()
        if not os.path.exists(r'IMAGES\ROI'):
            os.makedirs(r'IMAGES\ROI')
        else:
            self.output_path = rf'{os.getcwd()}\IMAGES\ROI\test_scene_{str(self.scene.split("_")[1])}.mp4'
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter(self.output_path, self.codec, 20, (self.screen_width, self.screen_height))

        self.lic_plate_frame_loc = rf'{os.getcwd()}\IMAGES\ROI\FRAME_LP-1.jpg'
        self.lic_plate_frame_loc2 = rf'{os.getcwd()}\IMAGES\ROI\FRAME_LP-2.jpg'
        self.lic_plate_frame_loc3 = rf'{os.getcwd()}\IMAGES\ROI\FRAME_LP-3.jpg'
        self.prev_entries = input("do you want to delete the previous entries?:\t")
        if self.prev_entries == 'y' or self.prev_entries == 'y'.upper():
            with open(f'scenes_json/{self.json_file}' if not self.scene == 'Scene_8'
                      else 'scenes_json/Scene_8.json', mode='r', encoding='utf-8') as file:
                export_dict = json.load(file)
                print(type(export_dict['Export']))
                if "LPR_Events" in export_dict['Export'].keys():
                    del export_dict['Export']['LPR_Events']
                    print('previous entries removed.')
                if not file.closed:
                    file.close()
                    print('file closed:', file.closed)
            with open(f'scenes_json/{self.json_file}' if not self.scene == 'Scene_8'
                      else 'scenes_json/Scene_8.json', mode='w', encoding='utf-8') as file:
                file.write(json.dumps(export_dict, indent=2))
                if not file.closed:
                    file.close()
                    print(f'file closed after modification: {file.closed}.')
        elif self.prev_entries == 'n' or self.prev_entries == 'n'.upper():
            self.process_anpr_pipeline()

    @staticmethod
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

    @staticmethod
    def extract_from_image_text(filepath: Union[AnyStr, List[AnyStr]]) -> str:
        client = boto3.client('textract', region_name="us-east-2")
        with open(filepath, 'rb') as imgfile:
            imgBytes = bytearray(imgfile.read())

        result = client.detect_document_text(Document={'Bytes': imgBytes})
        elements = result['Blocks']
        text_arr = [item['Text'] for item in elements if item['BlockType'] == "LINE"]

        for text in text_arr:
            return text

    @staticmethod
    def perform_multiple_json_process(lp_text: str, conf: Union[int, float], datetime_text: str,
                                      json_file: str, x: int, y: int, w: int, h: int) -> None:
        if lp_text is not None:
            result_dict = append_all_contents_2(text=lp_text, conf=conf, x=x, y=y, w=w, h=h, timestamp=str(datetime_text))
            print(result_dict)
            store_detected_events_in_corr_json(result_dict, json_file)

    def process_anpr_pipeline(self):
        try:
            with open(f'scenes_json/{self.json_file}' if not self.scene == 'Scene_8' else 'scenes_json/Scene_8.json', mode='r') as f1:
                print(f'file chosen:', f1.name.split("/")[-1] if "/" in f1.name else os.path.split(f1.name)[-1], 'for', self.scene)
                json_data = json.load(f1)
                sensor_dict = json_data['Export']['Sensors']
                for item in sensor_dict:
                    if item.get('EventName') == 'Stop Arm ST':
                        # bsa_start_date = str(item.get('DateTimeStart')).split(' ')[0]
                        bsa_start_time = str(item.get('DateTimeStart'))
                        bsa_stop_time = str(item.get('DateTimeStop'))
                        start_time = datetime.strptime(bsa_start_time, "%Y-%m-%d %H:%M:%S")
                        stop_time = datetime.strptime(bsa_stop_time, "%Y-%m-%d %H:%M:%S")

                        while True:
                            ret2, frame2 = self.video2_cap.read()
                            ret3, frame3 = self.video3_cap.read()
                            ret4, frame4 = self.video4_cap.read()

                            # frame resize
                            frame2 = cv2.resize(frame2, (int(self.screen_width / 2), int(self.screen_height / 2)))
                            frame3 = cv2.resize(frame3, (int(self.screen_width / 2), int(self.screen_height / 2)))
                            frame4 = cv2.resize(frame4, (int(self.screen_width / 2), int(self.screen_height / 2)))

                            x1, y1, x2, y2, conf = self.roboflow(frame2)
                            x31, y31, x32, y32, conf3 = self.roboflow(frame3)
                            x41, y41, x42, y42, conf4 = self.roboflow(frame4)

                            actual_time = start_time
                            print('actual_time', actual_time)

                            if start_time <= actual_time <= stop_time:
                                if (x1, y1, x2, y2) != (0, 0, 0, 0) and (x31, y31, x32, y32) != (0, 0, 0, 0)\
                                        and (x41, y41, x42, y42) != (0, 0, 0, 0):
                                    print('got the frames')
                                    print(f'detection in: {actual_time.time()}.')
                                    if np.less_equal(x1, frame2.shape[1]):
                                        if 35 <= y1 <= np.subtract(frame2.shape[0], 40):
                                            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.imwrite(self.lic_plate_frame_loc, frame2[y1: y2, x1: x2])

                                    if np.less_equal(x31, frame3.shape[1]):
                                        if 35 <= y31 <= np.subtract(frame3.shape[0], 40):
                                            cv2.rectangle(frame3, (x31, y31), (x32, y32), (255, 0, 0), 2)
                                            cv2.imwrite(self.lic_plate_frame_loc2, frame3[y31: y32, x31: x32])

                                    if np.less_equal(x41, frame4.shape[1]):
                                        if 35 <= y41 <= np.subtract(frame4.shape[0], 40):
                                            cv2.rectangle(frame4, (x41, y41), (x42, y42), (0, 0, 255), 2)
                                            cv2.imwrite(self.lic_plate_frame_loc3, frame4[y41: y42, x41: x42])

                                    frame_loc_1_stored = cv2.imread(self.lic_plate_frame_loc, flags=cv2.IMREAD_COLOR)
                                    frame_loc_2_stored = cv2.imread(self.lic_plate_frame_loc2, flags=cv2.IMREAD_COLOR)
                                    frame_loc_3_stored = cv2.imread(self.lic_plate_frame_loc3, flags=cv2.IMREAD_COLOR)

                                    frame_loc_1_scaled = cv2.resize(frame_loc_1_stored, (300, 200))
                                    frame_loc_2_scaled = cv2.resize(frame_loc_2_stored, (300, 200))
                                    frame_loc_3_scaled = cv2.resize(frame_loc_3_stored, (300, 200))

                                    stored_img_stacked = np.hstack((frame_loc_1_scaled, frame_loc_2_scaled, frame_loc_3_scaled))
                                    cv2.imshow('Violator Plates', stored_img_stacked)
                                    cv2.moveWindow('Frame Detected Texts', 0, np.floor_divide(self.screen_height, 2))
                                    print(f'frame 2 cropped portions saved to file: {os.path.split(self.lic_plate_frame_loc)[1]}.')
                                    print(f'frame 3 cropped portions saved to file: {os.path.split(self.lic_plate_frame_loc2)[1]}.')
                                    print(f'frame 4 cropped portions saved to file: {os.path.split(self.lic_plate_frame_loc3)[1]}.')

                                    cv2.imwrite(self.lic_plate_frame_loc, frame_loc_1_scaled)
                                    cv2.imwrite(self.lic_plate_frame_loc2, frame_loc_2_scaled)
                                    cv2.imwrite(self.lic_plate_frame_loc3, frame_loc_3_scaled)

                                    frame_2_text = self.extract_from_image_text(self.lic_plate_frame_loc)
                                    frame_3_text = self.extract_from_image_text(self.lic_plate_frame_loc2)
                                    frame_4_text = self.extract_from_image_text(self.lic_plate_frame_loc3)

                                    if frame_2_text is not None:
                                        cv2.putText(frame2, frame_2_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                    if frame_3_text is not None:
                                        cv2.putText(frame3, frame_3_text, (x31, y31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                                    if frame_4_text is not None:
                                        cv2.putText(frame4, frame_4_text, (x41, y41), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                                    self.perform_multiple_json_process(frame_2_text, conf, str(actual_time), self.json_file, x1, y1, x2, y2)
                                    self.perform_multiple_json_process(frame_3_text, conf3, str(actual_time), self.json_file, x31, y31, x32, y32)
                                    self.perform_multiple_json_process(frame_4_text, conf4, str(actual_time), self.json_file, x41, y41, x42, y42)
                                    print('Stop Arm: OPEN')
                                    start_time += timedelta(seconds=1)

                                else:
                                    print('Roboflow unable to detect..')
                                    print('Stop Arm: CLOSED')

                            row1 = np.hstack((frame3, frame2))
                            self.blank[0: int(self.screen_height / 2), 0: self.screen_width] = row1
                            self.blank[int(self.screen_height / 2): int(self.screen_height),
                            int(self.screen_width / 2): int(self.screen_width)] = frame4[:, :]
                            cv2.imshow('blank', self.blank)
                            cv2.namedWindow('blank', flags=cv2.WND_PROP_FULLSCREEN)
                            cv2.setWindowProperty('blank', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.moveWindow('blank', -20, 0)  # set to left
                            # cv2.moveWindow('blank', self.screen_width - 10, 50)  # set to right
                            self.out_video.write(self.blank)

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.video2_cap.release()
                                self.video3_cap.release()
                                self.video4_cap.release()
                                break

                        cv2.destroyAllWindows()

        except json.JSONDecodeError as e:
            print(e.msg)


anpr = AutomaticNumberPlateRecognition()
anpr.process_anpr_pipeline()
