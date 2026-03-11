from dataclasses import dataclass
import numpy as np
from collections import deque
import cv2
from yolox.tracker.byte_tracker import BYTETracker, STrack


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class Detection(object):
    def __init__(self, ltrb, track_id, is_break_in):
        self.track_id = track_id
        self.ltrb = ltrb
        self.is_break_in = is_break_in  # 是否闯入
        self.track_list = deque(maxlen=30)

    def update(self, ltrb, is_break_in):
        self.ltrb = ltrb
        self.is_break_in = is_break_in
        l, t, r, b = ltrb
        self.track_list.append(((l+r)//2, b))


class PersonTrack(object):
    def __init__(self):
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.detection_dict = {}

        # 防溺水电子警戒线（检测跨线行为）
        self.electronic_line = [(300, 400), (800, 400)]  # 第一条线（危险区域）

        # 防溺水人工警告线（仅提示，不触发警报）
        self.manual_line = [(300, 300), (800, 300)]  # 第二条线（警告区域）

        # 所有警戒线（用于可视化）
        self.warning_lines = [
            {"line": self.electronic_line, "color": (0, 0, 255), "name": "电子警戒线"},  # 红色
            {"line": self.manual_line, "color": (0, 255, 255), "name": "人工警告线"}  # 黄色
        ]

    def is_cross_line(self, prev_point, current_point, line):
        """
        判断是否跨过指定警戒线
        :param prev_point: 上一帧位置 (x, y)
        :param current_point: 当前帧位置 (x, y)
        :param line: 警戒线 [(x1, y1), (x2, y2)]
        :return: True（跨过） or False（未跨过）
        """
        (x1, y1), (x2, y2) = line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        prev_side = A * prev_point[0] + B * prev_point[1] + C
        current_side = A * current_point[0] + B * current_point[1] + C

        # 从下往上跨过线
        if prev_side < 0 and current_side > 0:
            return True
        return False

    def update_track(self, boxes, frame):
        tracks = self.byte_tracker.update(
            output_results=boxes,
            img_info=frame.shape,
            img_size=frame.shape
        )

        new_detection_dict = {}
        for track in tracks:
            l, t, r, b = track.tlbr.astype(np.int32)
            track_id = track.track_id
            current_point = ((l + r) // 2, b)  # 当前帧脚部位置

            # 初始化闯入状态
            is_crossed_electronic = False  # 是否跨过电子警戒线
            is_crossed_manual = False  # 是否跨过人工警告线

            # 检查是否跨过电子警戒线
            if track_id in self.detection_dict:
                prev_point = self.detection_dict[track_id].track_list[-1]
                is_crossed_electronic = self.is_cross_line(prev_point, current_point, self.electronic_line)

            # 检查是否跨过人工警告线（可选）
            if track_id in self.detection_dict:
                prev_point = self.detection_dict[track_id].track_list[-1]
                is_crossed_manual = self.is_cross_line(prev_point, current_point, self.manual_line)

            # 更新 Detection 对象
            if track_id in self.detection_dict:
                detection = self.detection_dict[track_id]
                detection.update((l, t, r, b), is_crossed_electronic)  # 仅电子警戒线触发警报
            else:
                detection = Detection((l, t, r, b), track_id, is_crossed_electronic)

            new_detection_dict[track_id] = detection

        self.detection_dict = new_detection_dict
        return self.detection_dict, self.warning_lines  # 返回所有警戒线