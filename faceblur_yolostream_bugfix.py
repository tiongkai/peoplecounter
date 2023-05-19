#tracktest environment
import os
import cv2
HOME = os.getcwd()
print(HOME)
name = 'final'
TARGET_VIDEO_PATH = f"{HOME}/"+name+"-result.mp4"
SOURCE_VIDEO_PATH =name+'.mp4'#'rtsp://:8554/stream'#0

import supervision
print("supervision.__version__:", supervision.__version__)

from supervision.draw.color import ColorPalette
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List
import numpy as np

import ultralytics
ultralytics.checks()
import sys
sys.path.append(f"{HOME}/ByteTrack")


import yolox
print("yolox.__version__:", yolox.__version__)
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

cap = cv2.VideoCapture(name+'.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
size = (frame_width, frame_height)



MODEL = "yolov8n-face.pt"
result = cv2.VideoWriter(name+'Resultv8n.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, size)
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - [people]
CLASS_ID = [0]

# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())

width=640
height=480
fps=30

total_frames=-1
# create VideoInfo instance
video_info = VideoInfo(width, height, fps, total_frames)

# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create LineCounter instance

# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=.5)
line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=.5)

# open target video file
with VideoSink(TARGET_VIDEO_PATH,video_info) as sink:
    # loop over video frames
    for frame in generator:
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # updating line counter
        ''''''
        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(detections):
            x1, y1, x2, y2 = xyxy.astype(int)
            roi_color = frame[int(y1):int(y2),int(x1):int(x2)]
            blur = cv2.GaussianBlur(roi_color, (101,101), 0) 
            frame[int(y1):int(y2),int(x1):int(x2)]=blur

    # Display the resulting frame
        cv2.imshow('Frame', frame)
        result.write(frame)
        #print(width,height,fps,total_frames)
          
    # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break