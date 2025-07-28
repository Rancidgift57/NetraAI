import cv2
import numpy as np
from enum import Enum
from collections import Counter
from typing import List
import os
import torch
import time
import pyttsx3
import threading
import onnxruntime as ort

class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_HYBRID = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"

class Midas:
    def __init__(self, modelType: ModelType = ModelType.DPT_HYBRID, model_path: str = "midas_small.onnx"):
        try:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.modelType = modelType
        except Exception as e:
            print(f"Error loading MiDaS ONNX model: {e}")
            raise

    def transform(self):
        # Preprocessing pipeline for MiDaS ONNX (384x384 input)
        self.transform = Compose([
            Resize(384, 384),  # Resize to 384x384 for the ONNX model
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

    def predict(self, frame):
        # Convert frame to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        input_data = self.transform(img)
        
        # Run inference
        prediction = self.session.run(None, {self.input_name: input_data})[0]
        prediction = prediction.squeeze()  # Remove batch dimension
        
        # Resize to original frame size
        input_height, input_width = frame.shape[:2]
        prediction = cv2.resize(prediction, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to torch tensor for compatibility with existing code
        raw_depth = torch.tensor(prediction, device='cpu')
        
        print(f"Raw depth min: {raw_depth.min().item():.2f}, max: {raw_depth.max().item():.2f}, mean: {raw_depth.mean().item():.2f}")
        
        # Scale depth values
        DEPTH_SCALE_FACTOR = 0.002
        depth_scaled = raw_depth * DEPTH_SCALE_FACTOR
        depth_scaled = torch.clamp(depth_scaled, min=0.1, max=100.0)
        
        # Normalize for visualization
        depth_map = cv2.normalize(depth_scaled.numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        
        return depth_map, depth_scaled

# Preprocessing classes for ONNX models
class Resize:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        return img

class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = image / 255.0
        image = (image - self.mean) / self.std
        return image

class PrepareForNet:
    def __call__(self, image):
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image.astype(np.float32)

# Compose class for preprocessing pipeline
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class YOLOv8:
    def __init__(self, model_path: str = "yolov8s.onnx"):
        try:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            # COCO dataset class names (80 classes)
            self.names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            self.num_classes = len(self.names)
            self.conf_thres = 0.3
            self.iou_thres = 0.45
        except Exception as e:
            print(f"Error loading YOLO ONNX model: {e}")
            raise

    def useCUDA(self):
        print("YOLO ONNX model using CPUExecutionProvider")

    def non_max_suppression(self, boxes, scores, iou_thres):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        
        return keep

    def detect_objects(self, frame):
        height = frame.shape[0]
        roi = frame[int(height * 0.4):, :]
        
        # Preprocess for YOLO (640x640 input)
        transform = Compose([
            Resize(640, 640),
            NormalizeImage(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            PrepareForNet()
        ])
        input_data = transform(roi)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})[0]
        print(f"YOLO output shape: {outputs.shape}")  # Debug output shape
        
        # Transpose to [1, num_boxes, num_outputs]
        outputs = outputs.transpose((0, 2, 1))
        print(f"Transposed output shape: {outputs.shape}")  # Debug transposed shape
        
        # Extract boxes, scores, and classes
        boxes = outputs[0, :, :4]  # [x_center, y_center, width, height]
        scores = outputs[0, :, 4:4+self.num_classes]  # Class probabilities
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Debug class IDs
        print(f"Class IDs: {class_ids}")
        print(f"Confidences: {confidences}")
        
        # Filter by confidence and valid class IDs
        mask = (confidences > self.conf_thres) & (class_ids < self.num_classes)
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            print("No valid detections after filtering")
            return None, torch.empty((0, 4), device='cpu')
        
        # Convert boxes to [x1, y1, x2, y2]
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_center to x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_center to y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x1 to x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y1 to y2
        
        # Scale boxes back to ROI size
        roi_height, roi_width = roi.shape[:2]
        boxes[:, [0, 2]] *= roi_width / 640
        boxes[:, [1, 3]] *= roi_height / 640
        
        # Apply NMS
        indices = self.non_max_suppression(boxes, confidences, self.iou_thres)
        boxes = boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]
        
        # Adjust boxes to original frame coordinates
        boxes[:, [1, 3]] += int(height * 0.4)
        
        # Create results object for compatibility
        class Results:
            def __init__(self):
                self.boxes = Boxes()
        
        class Boxes:
            def __init__(self):
                self.xyxy = torch.tensor(boxes, device='cpu')
                self.conf = torch.tensor(confidences, device='cpu')
                self.cls = torch.tensor(class_ids, device='cpu')
        
        results = Results()
        adjusted_boxes = torch.tensor(boxes, device='cpu')
        
        return results, adjusted_boxes

    def draw_detections(self, frame, results, adjusted_boxes, raw_depth):
        if results is None or len(adjusted_boxes) == 0:
            return frame
        for i, box in enumerate(adjusted_boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = results.boxes.conf[i]
            cls = int(results.boxes.cls[i])
            if cls >= len(self.names):  # Additional safety check
                print(f"Warning: Invalid class ID {cls}, skipping detection")
                continue
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
    def __init__(self, modelType: ModelType, midas_model_path: str = "midas_small.onnx", yolo_model_path: str = "yolov8s.onnx"):
        self.midas = Midas(modelType, midas_model_path)
        self.yolo = YOLOv8(yolo_model_path)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.last_audio_time = 0
        self.audio_cooldown = 2
        self.audio_thread = None
        self.zones = None
        self.frame_skip = 2
        self.frame_count = 0
        self.fps_log = []
        self.stop_if_no_objects_but_close = True
        self.stop_count = 0
        self.last_guidance = ""

    def useCUDA(self):
        self.yolo.useCUDA()
        print("MiDaS ONNX model using CPUExecutionProvider")

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

    def process_video(self, video_path, output_width=640, output_height=480):
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist")
            return

        print(f'Processing video: {video_path} (press q to quit)...')
        capObj = cv2.VideoCapture(video_path)
        if not capObj.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        total_frames = int(capObj.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capObj.get(cv2.CAP_PROP_FPS)
        print(f"Total frames: {total_frames}, FPS: {fps}")

        self.zones = self.adjust_zones(640, output_width)

        dummy_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        self.midas.predict(dummy_frame)
        self.yolo.detect_objects(dummy_frame)
        print("Models warmed up")

        last_detections = None
        last_results = None
        last_depth = None
        start_time = time.time()

        while True:
            ret, frame = capObj.read()
            if not ret or frame is None:
                print("Frame read error: Exiting...")
                break

            self.frame_count += 1
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_fps = self.frame_count / elapsed_time
                self.fps_log.append(current_fps)
                print(f"Processing frame {self.frame_count}/{total_frames}, FPS: {current_fps:.2f}", end='\r')

            if self.frame_count % self.frame_skip == 0:
                depth_map, raw_depth = self.midas.predict(frame)
                results, adjusted_boxes = self.yolo.detect_objects(frame)
                last_detections = adjusted_boxes
                last_results = results
                last_depth = raw_depth
            else:
                adjusted_boxes = last_detections if last_detections is not None else torch.empty((0, 4), device='cpu')
                results = last_results
                raw_depth = last_depth if last_depth is not None else torch.zeros((output_height, output_width), device='cpu')
                depth_map = cv2.normalize(raw_depth.numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

            detected_frame = self.yolo.draw_detections(frame.copy(), results, adjusted_boxes, raw_depth)
            detected_frame = self.draw_zones(detected_frame, self.zones)
            navigation_text = self.analyze_navigation(results, adjusted_boxes, raw_depth, detected_frame, self.zones)

            if navigation_text:
                self.speak_guidance(navigation_text)

            combined = np.hstack((detected_frame, depth_map))
            cv2.imshow('YOLO + MiDaS Navigation', combined)

            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

        avg_fps = sum(self.fps_log) / len(self.fps_log) if self.fps_log else 0
        print(f"\nAverage FPS: {avg_fps:.2f}")

        capObj.release()
        cv2.destroyAllWindows()

def run():
    navSystem = NavigationSystem(
        ModelType.DPT_HYBRID,
        midas_model_path="midas_small.onnx",
        yolo_model_path="yolov8s.onnx"
    )
    navSystem.useCUDA()
    navSystem.transform()
    navSystem.process_video("RightLeft.mp4", output_width=640, output_height=480)

if __name__ == '__main__':
    run()