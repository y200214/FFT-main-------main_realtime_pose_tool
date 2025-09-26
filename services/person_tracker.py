# ファイル名: services/person_tracker.py

import cv2
import logging
from ultralytics import YOLO
from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
from constants import REALTIME_ID_PREFIX

logger = logging.getLogger(__name__)

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.rects = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect):
        self.objects[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.rects[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.rects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.rects[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        return self.rects

class PersonTracker:
    """YOLOv8を使い、フレーム内の顔を検出・追跡するクラス。"""

    def __init__(self, model_path, device='cpu'):
        logger.info(f"YOLOv8顔検出モデル '{model_path}' をデバイス '{device}' で読み込んでいます...")
        self.model = YOLO(model_path)
        self.device = device
        self.tracker = CentroidTracker(maxDisappeared=80)
        logger.info("YOLOv8モデルの読み込みが完了しました。")

    def track(self, frame):
        results = self.model.predict(frame, verbose=False, conf=0.6, iou=0.4)
        rects = [box.xyxy[0].cpu().numpy().astype(int) for r in results for box in r.boxes]
        
        tracked_objects = self.tracker.update(rects)
        
        tracked_persons = []
        for (track_id, box) in tracked_objects.items():
            tracked_persons.append({
                "id": f"{REALTIME_ID_PREFIX}{track_id}",
                "box": box
            })
        return tracked_persons, frame