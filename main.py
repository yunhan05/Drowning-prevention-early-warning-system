import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
print("0")
from mask_face import mask_face
print("2")
from track import PersonTrack

print("1")
def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), tex_size=30):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_style = ImageFont.truetype(
        "./fonts/MSYH.ttc", tex_size, encoding="utf-8")
    draw.text(position, text, text_color, font=font_style)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
print("2")
class BreakInDetection:
    def __init__(self):
        self.yolov5_model = torch.hub.load('yolov5'
                                           , 'custom'
                                           , path='./weights/yolov5s.pt'
                                           , source='local')
        self.yolov5_model.conf = 0.7
        self.tracker = PersonTrack()

    @staticmethod
    def yolo_pd_to_numpy(yolo_pd):
        box_list = yolo_pd.to_numpy()
        detections = []
        for box in box_list:
            l, t = int(box[0]), int(box[1])
            r, b = int(box[2]), int(box[3])

            conf = box[4]

            detections.append([l, t, r, b, conf])
        return np.array(detections, dtype=float)

    def plot_detection(self, person_track_dict, warning_lines, frame, frame_idx):
        break_in_num = 0
        for track_id, detection in person_track_dict.items():
            l, t, r, b = detection.ltrb
            track_id = detection.track_id

            if detection.is_break_in:  # 跨过电子警戒线
                box_color = (0, 0, 255)  # 红色框
                id_color = (0, 0, 255)  # 红色ID
                break_in_num += 1
            else:
                box_color = (0, 255, 0)  # 绿色框
                id_color = (255, 0, 0)  # 蓝色ID

            frame[t:b, l:r] = mask_face(frame[t:b, l:r])
            cv2.rectangle(frame, (l, t), (r, b), box_color, 1)
            cv2.putText(frame, f'id-{track_id}', (l + 2, t - 3), cv2.FONT_HERSHEY_PLAIN, 3, id_color, 2)

        # 绘制所有警戒线
        for line_info in warning_lines:
            (x1, y1), (x2, y2) = line_info["line"]
            color = line_info["color"]
            name = line_info["name"]

            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            frame = cv2_add_chinese_text(frame, name, (x1 + 50, y1 - 20), color, 30)

        # 统计信息
        info_frame = np.zeros((200, 400, 3), np.uint8)
        if_l, if_t = 100, 100
        if_r, if_b = if_l + 400, if_t + 200
        frame_part = frame[if_t:if_b, if_l:if_r]
        mixed_frame = cv2.addWeighted(frame_part, 0.6, info_frame, 0.7, 0)
        frame[if_t:if_b, if_l:if_r] = mixed_frame

        frame = cv2_add_chinese_text(frame, f'统计', (if_l + 150, if_t + 10), (255, 0, 0), 40)
        frame = cv2_add_chinese_text(frame, f'当前闯入电子警戒线 {break_in_num} 人', (if_l + 60, if_t + 80),
                                     (255, 0, 0), 35)

        return frame

    def detect(self):
        cap = cv2.VideoCapture(r"C:\Users\张韵涵\Desktop\5250f641cd5ae24249f18fee175934b2.mp4")
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        print(fps)
        video_writer = cv2.VideoWriter('./video_result.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (video_w, video_h))


        frame_idx = 0
        while cap.isOpened():
            frame_idx += 1
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            results = self.yolov5_model(frame[:, :, ::-1])
            pd = results.pandas().xyxy[0]
            person_pd = pd[pd['name'] == 'person']
            person_det_boxes = self.yolo_pd_to_numpy(person_pd)
            if len(person_det_boxes) > 0:
                person_track_dict, penalty_zone_point_list = self.tracker.update_track(person_det_boxes, frame)
                frame = self.plot_detection(person_track_dict, penalty_zone_point_list, frame, frame_idx)

            cv2.imshow('Break in Detection', frame)
            video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

print("3")
if __name__ == '__main__':
    BreakInDetection().detect()
