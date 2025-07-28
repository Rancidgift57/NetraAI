import cv2
import torch
import numpy as np
from enum import Enum
from ultralytics import YOLO
from collections import Counter
from typing import List
import os
import time
import pyttsx3
import threading
import queue

class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_HYBRID = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"

class Midas:
    def __init__(self, modelType: ModelType = ModelType.DPT_HYBRID):
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", modelType.value)
            self.modelType = modelType
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            raise

    def useCUDA(self):
        self.device = torch.device("cuda")
        print(f'Using {self.device} for MiDaS')
        self.midas.to(self.device)
        self.midas.eval()

    def transform(self):
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform if self.modelType.value in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()

        raw_depth = prediction
        print(f"Raw depth min: {raw_depth.min().item():.2f}, max: {raw_depth.max().item():.2f}, mean: {raw_depth.mean().item():.2f}")
        
        DEPTH_SCALE_FACTOR = 0.002
        depth_scaled = raw_depth * DEPTH_SCALE_FACTOR
        depth_scaled = torch.clamp(depth_scaled, min=0.1, max=100.0)

        depth_map = cv2.normalize(depth_scaled.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        return depth_map, depth_scaled

class YOLOv8:
    def __init__(self):
        try:
            self.model = YOLO("yolov8s.pt")
            self.names = self.model.names
            self.model.conf = 0.3
            self.model.iou = 0.45
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def useCUDA(self):
        self.device = torch.device("cuda")
        print(f'Using {self.device} for YOLO')
        self.model.to(self.device)

    def detect_objects(self, frame):
        height = frame.shape[0]
        roi = frame[int(height * 0.4):, :]
        results = self.model(roi)

        result = results[0]
        boxes = result.boxes.xyxy.clone()
        boxes[:, [1, 3]] += int(height * 0.4)
        return result, boxes

    def draw_detections(self, frame, results, adjusted_boxes, raw_depth):
        if results is None:
            return frame
        for i, box in enumerate(adjusted_boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = results.boxes.conf[i]
            cls = int(results.boxes.cls[i])
            depth_region = raw_depth[y1:y2, x1:x2]
            depth_value = torch.mean(depth_region).item()
            if depth_value <= 7:
                label = f"{self.names[cls]} {conf:.2f} ({depth_value:.2f}m)"
                color = (0, 255, 0) if depth_value >= 7 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

class NavigationSystem:
    def __init__(self, modelType: ModelType):
        self.midas = Midas(modelType)
        self.yolo = YOLOv8()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.last_audio_time = 0
        self.audio_cooldown = 2
        self.audio_thread = None
        self.zones = None
        self.frame_skip = 2  # Initial frame skip value
        self.frame_count = 0
        self.fps_log = []
        self.stop_if_no_objects_but_close = True
        self.stop_count = 0
        self.last_guidance = ""
        self.frame_queue = queue.Queue(maxsize=10)  # Increased from 5 to 10 for better buffering

    def useCUDA(self):
        self.midas.useCUDA()
        self.yolo.useCUDA()

    def transform(self):
        self.midas.transform()

    def adjust_zones(self, original_width, target_width):
        scale = target_width / original_width
        return {
            'LEFT': (0, int(213 * scale)),
            'CENTER': (int(214 * scale), int(426 * scale)),
            'RIGHT': (int(427 * scale), target_width)
        }

    def draw_zones(self, frame, zones):
        h, w, _ = frame.shape
        cv2.line(frame, (zones['LEFT'][1], 0), (zones['LEFT'][1], h), (255, 255, 0), 1)
        cv2.line(frame, (zones['CENTER'][1], 0), (zones['CENTER'][1], h), (255, 255, 0), 1)
        cv2.putText(frame, "LEFT", (zones['LEFT'][0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "CENTER", (zones['CENTER'][0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "RIGHT", (zones['RIGHT'][0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return frame

    def calculate_clearance(self, depth, zones):
        left_depth = torch.mean(depth[:, zones['LEFT'][0]:zones['LEFT'][1]]).item()
        center_depth = torch.mean(depth[:, zones['CENTER'][0]:zones['CENTER'][1]]).item()
        right_depth = torch.mean(depth[:, zones['RIGHT'][0]:zones['RIGHT'][1]]).item()
        return left_depth, center_depth, right_depth

    def analyze_navigation(self, results, adjusted_boxes, depth, frame, zones):
        h, w, _ = frame.shape
        left_clear, center_clear, right_clear = True, True, True
        objects_detected = []
        obstacle_too_close = False

        left_depth, center_depth, right_depth = self.calculate_clearance(depth, zones)

        if results is not None and len(adjusted_boxes) > 0:
            for i, box in enumerate(adjusted_boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = results.boxes.conf[i]
                cls = int(results.boxes.cls[i])

                if conf < 0.5:
                    continue

                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                if x2 <= x1 or y2 <= y1:
                    continue

                depth_region = depth[y1:y2, x1:x2]
                if depth_region.size == 0 or torch.isnan(depth_region).any():
                    continue

                obj_depth = torch.mean(depth_region).item()

                if obj_depth <= 1:
                    obstacle_too_close = True
                    print(f"Object {self.yolo.names[cls]} at {obj_depth:.2f}m is closer than 1m")

                if obj_depth > 7:
                    print(f"Ignoring object {self.yolo.names[cls]} at {obj_depth:.2f}m (beyond 7m threshold)")
                    continue

                object_name = self.yolo.names[cls]
                objects_detected.append((object_name, obj_depth, x_center))

                if zones['LEFT'][0] <= x_center <= zones['LEFT'][1]:
                    left_clear = False
                    print(f"Object {object_name} at {obj_depth:.2f}m in LEFT zone")
                elif zones['CENTER'][0] <= x_center <= zones['CENTER'][1]:
                    center_clear = False
                    print(f"Object {object_name} at {obj_depth:.2f}m in CENTER zone")
                elif zones['RIGHT'][0] <= x_center <= zones['RIGHT'][1]:
                    right_clear = False
                    print(f"Object {object_name} at {obj_depth:.2f}m in RIGHT zone")

        if not objects_detected:
            print(f"No objects detected within 7m, Left depth: {left_depth:.2f}m, Center depth: {center_depth:.2f}m, Right depth: {right_depth:.2f}m")
            if self.stop_if_no_objects_but_close:
                left_clear = left_depth > 1.8
                center_clear = center_depth > 1.8
                right_clear = right_depth > 1.8

        cv2.putText(frame, f"Left Path: {'Clear' if left_clear else 'Blocked'} ({left_depth:.2f}m)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if left_clear else (0, 0, 255), 2)
        cv2.putText(frame, f"Center Path: {'Clear' if center_clear else 'Blocked'} ({center_depth:.2f}m)",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if center_clear else (0, 0, 255), 2)
        cv2.putText(frame, f"Right Path: {'Clear' if right_clear else 'Blocked'} ({right_depth:.2f}m)",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if right_clear else (0, 0, 255), 2)

        empty_paths = []
        if left_clear:
            empty_paths.append("Left")
        if center_clear:
            empty_paths.append("Center")
        if right_clear:
            empty_paths.append("Right")

        if len(empty_paths) == 3:
            navigation_text = "Path clear"
            cv2.putText(frame, "Path Clear", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            print("Advising: Path clear")
        elif len(empty_paths) > 0:
            if obstacle_too_close:
                cv2.putText(frame, "Obstacle Close!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if "Left" in empty_paths:
                navigation_text = "Move Left"
                cv2.putText(frame, "Move Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print("Advising: Move Left")
            elif "Right" in empty_paths:
                navigation_text = "Move Right"
                cv2.putText(frame, "Move Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print("Advising: Move Right")
            else:
                navigation_text = "Move Center"
                cv2.putText(frame, "Move Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print("Advising: Move Center")
        else:
            navigation_text = "Stop, all paths blocked"
            if obstacle_too_close:
                navigation_text = "Stop, obstacle too close"
                cv2.putText(frame, "STOP - Too Close", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                cv2.putText(frame, "STOP - All Paths Blocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print(f"Advising: {navigation_text}")

        if navigation_text.startswith("Stop"):
            if self.last_guidance.startswith("Stop"):
                self.stop_count += 1
            else:
                self.stop_count = 1
        else:
            self.stop_count = 0

        self.last_guidance = navigation_text

        if self.stop_count >= 3:
            check_text = "CHECK WITH YOUR STICK OR HAND"
            cv2.putText(frame, check_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            navigation_text = check_text
            print("Advising: CHECK WITH YOUR STICK OR HAND")

        return navigation_text

    def _play_audio(self, guidance_text):
        self.engine.say(guidance_text)
        self.engine.runAndWait()

    def speak_guidance(self, guidance_text):
        current_time = time.time()
        if guidance_text in ["STOP", "Move Left", "Move Right", "Move Center"] or guidance_text.startswith("Stop") or guidance_text == "CHECK WITH YOUR STICK OR HAND" or (current_time - self.last_audio_time >= self.audio_cooldown):
            if self.audio_thread is None or not self.audio_thread.is_alive():
                self.audio_thread = threading.Thread(target=self._play_audio, args=(guidance_text,))
                self.audio_thread.start()
                self.last_audio_time = current_time

    def capture_frames(self, output_width, output_height):
        stream_url = 1  # Update this with your IVCam URL
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"Error: Could not open stream {stream_url}")
            return
        
        print(f"Successfully opened stream {stream_url}")
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Frame dropped or stream interrupted")
                time.sleep(0.1)  # Brief pause before retrying
                continue
            
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                print("Frame queue full, dropping frame")
                pass  # Drop frame if queue is full
            
            time.sleep(0.01)  # Small delay to prevent overwhelming the queue
        
        cap.release()

    def process_realtime(self, output_width=640, output_height=480, target_fps=15):
        print(f'Starting real-time navigation (press q to quit)...')
        
        # Start frame capture in a separate thread
        capture_thread = threading.Thread(target=self.capture_frames, args=(output_width, output_height))
        capture_thread.daemon = True
        capture_thread.start()

        self.zones = self.adjust_zones(640, output_width)

        # Warm up GPU
        dummy_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        self.midas.predict(dummy_frame)
        self.yolo.detect_objects(dummy_frame)
        print("GPU warmed up")

        last_detections = None
        last_results = None
        last_depth = None
        start_time = time.time()
        frame_interval = 1.0 / target_fps  # Desired frame processing interval

        while True:
            try:
                frame = self.frame_queue.get(timeout=2.0)  # Increased timeout from 1.0 to 2.0
            except queue.Empty:
                print("No frame received from stream ")
                continue

            self.frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_fps = self.frame_count / elapsed_time
                self.fps_log.append(current_fps)
                cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Adaptive frame skipping based on FPS
                if current_fps < target_fps * 0.8:  # If FPS drops below 80% of target
                    self.frame_skip = min(self.frame_skip + 1, 5)  # Increase skip, cap at 5
                elif current_fps > target_fps * 1.2:  # If FPS exceeds 120% of target
                    self.frame_skip = max(self.frame_skip - 1, 1)  # Decrease skip, min 1

            if self.frame_count % self.frame_skip == 0:
                with torch.no_grad():
                    depth_map, raw_depth = self.midas.predict(frame)
                    results, adjusted_boxes = self.yolo.detect_objects(frame)
                    last_detections = adjusted_boxes
                    last_results = results
                    last_depth = raw_depth
            else:
                adjusted_boxes = last_detections if last_detections is not None else torch.empty((0, 4), device=self.midas.device)
                results = last_results
                raw_depth = last_depth if last_depth is not None else torch.zeros((output_height, output_width), device=self.midas.device)
                depth_map = cv2.normalize(raw_depth.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

            detected_frame = self.yolo.draw_detections(frame.copy(), results, adjusted_boxes, raw_depth)
            detected_frame = self.draw_zones(detected_frame, self.zones)
            navigation_text = self.analyze_navigation(results, adjusted_boxes, raw_depth, detected_frame, self.zones)

            if navigation_text:
                self.speak_guidance(navigation_text)

            combined = np.hstack((detected_frame, depth_map))
            cv2.imshow('YOLO + MiDaS Real-time Navigation', combined)

            # Control frame rate
            processing_time = time.time() - start_time - elapsed_time
            sleep_time = max(0, frame_interval - processing_time)
            time.sleep(sleep_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        avg_fps = sum(self.fps_log) / len(self.fps_log) if self.fps_log else 0
        print(f"\nAverage FPS: {avg_fps:.2f}")
        cv2.destroyAllWindows()

def run():
    navSystem = NavigationSystem(ModelType.DPT_HYBRID)
    navSystem.useCUDA()
    navSystem.transform()
    navSystem.process_realtime(output_width=640, output_height=480, target_fps=15)

if __name__ == '__main__':
    run()