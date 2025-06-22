import cv2
import torch
import os
import matplotlib.pyplot as plt
from moviepy.editor import *
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import logging
# from profiler import Profiler
import multiprocessing


@dataclass
class CropperConfig:
    sampling_rate_default: int = 30
    sampling_rate_tight: int = 5
    margin_ratio: float = 1.6
    square_mode: bool = True
    do_pose_estimation: bool = False
    show_object_detection: bool = False
    save_detection: bool = True
    model_version: int = 11
    confidence_threshold: float = 0.3
    
class VideoCropper:
    def __init__(self, config: Optional[CropperConfig] =  None):
        self.config = config or CropperConfig()
        self.model = None
        self.device = self._get_device()
        self.detector = None
        self.processor = None
        
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
    def load_model(self):
        if self.config.model_version == 8:
            self.model = YOLO("yolov8s.pt")
        elif self.config.model_version == 5:
            self.model = YOLO("yolov5s.pt")
            self.model.conf = self.config.confidence_threshold
            self.model.classes = [0]
        elif self.config.model_version == 11:
            self.model = YOLO("yolo11n.pt")
        
        self.model.fuse()
        self.model.to(self.device)
        #self.model.to("cpu")
        self.detector = ObjectDetector(self.model, self.config)
        
    def process_video(self, input_path: str, output_path: str, job_status: dict, jobid: str) -> bool:
        self.processor = VideoProcessor(input_path, output_path, job_status, jobid, self.config)
        self.processor.process_video(self.detector)
        return True

class ObjectDetector:
    def __init__(self, model: YOLO, config: CropperConfig):
        self.model = model
        self.config = config
        self.w_margin_list: List[float] = []
        self.h_margin_list: List[float] = []
        
    def detect_person(self, frame: np.ndarray, prev_center_x: float, prev_center_y: float, frame_index: int) -> Tuple[float, float]:
        # 프레임 너비 기반 탐지 가능한 중심 영역 정의
        frame_width = frame.shape[1]
        focus_zone_start = frame_width // 6
        focus_zone_end = frame_width * 5 // 6
        frame_center_x = frame_width // 2

        # 객체 탐지 실행
        results = self.model(frame, stream=True, verbose=False)
        selected_center_x, selected_center_y = -1, -1
        min_distance_to_center = float('inf')

        person_detected = False

        # 탐지 결과 반복
        for result in results:
            for bbox in result.boxes:
                # 바운딩 박스 좌표 및 클래스 정보 추출
                x1, y1, x2, y2 = map(round, bbox.xyxy[0].tolist())
                class_id = int(bbox.cls[0].item())
                confidence = bbox.conf[0].item()

                if class_id == 0:  # 사람이 탐지된 경우
                    logging.info(f"Detected person at {(x1, y1, x2, y2)} with confidence {confidence}")

                    # 사람 중심 좌표 계산
                    person_center_x = (x1 + x2) // 2
                    person_center_y = (y1 + y2) // 2

                    # 이전 중심 좌표와의 거리 계산
                    dist_to_prev = np.hypot(person_center_x - prev_center_x, person_center_y - prev_center_y)
                    dist_to_frame_center = abs(person_center_x - frame_center_x)

                    # 탐지 범위 바깥에 있거나 너무 튄 경우 무시
                    if not (focus_zone_start <= person_center_x <= focus_zone_end):
                        continue
                    if dist_to_prev > frame_width // 4:
                        continue

                    # 중심에 가장 가까운 사람 하나만 선택
                    if dist_to_frame_center < min_distance_to_center:
                        min_distance_to_center = dist_to_frame_center
                        selected_center_x = person_center_x
                        selected_center_y = person_center_y

                        # 마진 계산 및 저장
                        width_margin = ((x2 - x1) * self.config.margin_ratio) // 2
                        height_margin = ((y2 - y1) * self.config.margin_ratio) // 2
                        self.w_margin_list.append(width_margin)
                        self.h_margin_list.append(height_margin)

                        person_detected = True

                        # 옵션에 따라 탐지 이미지 저장
                        if self.config.save_detection:
                            annotated = result.plot()
                            output_path = f"temp/person_detected_{frame_index}.jpg"
                            cv2.imwrite(output_path, annotated)
                            logging.info(f"Detection saved to {output_path}")
                            img = annotated

            # 사람이 탐지되지 않은 경우에도 탐지 결과 저장 (옵션일 때만)
            if self.config.save_detection and not person_detected:
                annotated = result.plot()
                fallback_path = f"temp/detection_output_{frame_index}.jpg"
                realtime_output = f"detection_output.jpg"
                cv2.imwrite(fallback_path, annotated)
                cv2.imwrite(realtime_output, annotated)
                logging.info(f"Fallback detection saved to {fallback_path}")
                img = annotated

        # 탐지 실패 시 이전 중심 유지
        if selected_center_x == -1 or selected_center_y == -1:
            selected_center_x, selected_center_y = prev_center_x, prev_center_y

        return selected_center_x, selected_center_y, img
class TrajectorySmoothing:
    @staticmethod
    def centered_moving_average(data: List[float], window_size: int = 10) -> List[float]:
        avg_list = []
        for i in range(len(data)):
            start = max(0, i - window_size)
            end = min(len(data), i + window_size)
            avg = np.mean(data[start:end])
            avg_list.append(avg)
        return avg_list

class VideoProcessor:
    def __init__(self, input_path: str, output_path: str, job_status: dict, jobid: str, config: CropperConfig):
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.frames = []
        self.center_x_list = []
        self.center_y_list = []
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.job_status = job_status
        self.job_id = jobid
        # self.profiler = Profiler()
        
    def _init_video_capture(self):
        self.cap = cv2.VideoCapture(self.input_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.job_status[self.job_id] = {
            "status": "processing",
            "progress": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": self.input_path,
            "output_file": self.output_path,
        }
    def _crop_frame(self, frame: np.ndarray, center_x: float, center_y: float, 
                   w_margin: float, h_margin: float) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        
        crop_x1 = int(max(center_x - w_margin, 0))
        crop_y1 = int(max(center_y - h_margin, 0))
        crop_x2 = int(min(center_x + w_margin, frame_width))
        crop_y2 = int(min(center_y + h_margin, frame_height))
        
        if crop_x2 - crop_x1 < 2 * w_margin:
            if crop_x1 == 0:
                crop_x2 = crop_x1 + int(2 * w_margin)
            else:
                crop_x1 = crop_x2 - int(2 * w_margin)
                
        if crop_y2 - crop_y1 < 2 * h_margin:
            if crop_y1 == 0:
                crop_y2 = crop_y1 + int(2 * h_margin)
            else:
                crop_y1 = crop_y2 - int(2 * h_margin)
        
        blank_frame = np.zeros((int(h_margin * 2), int(w_margin * 2), 3), dtype=np.uint8)
        cropped_part = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_h, cropped_w = cropped_part.shape[:2]
        
        start_y = (blank_frame.shape[0] - cropped_h) // 2
        start_x = (blank_frame.shape[1] - cropped_w) // 2
        blank_frame[start_y:start_y + cropped_h, start_x:start_x + cropped_w] = cropped_part
        
        return cv2.cvtColor(blank_frame, cv2.COLOR_RGB2BGR)
    
    def process_video(self, detector: ObjectDetector):
        self._init_video_capture()
        logging.info(f"Processing video: {self.input_path}")
        logging.info(f"Frame size: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
        
        os.makedirs(f"temp/{self.job_id}", exist_ok=True)
        
        pre_center_x, pre_center_y = -1, -1
        frame_count = 0
        start_time = time.time()
        
        # self.profiler.start()
        # First pass: Detect person and collect center points
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_count % self.config.sampling_rate_default == 0 or frame_count < 90:
                center_x, center_y, img = detector.detect_person(frame, pre_center_x, pre_center_y, frame_count)
                # self.profiler.log_memory_usage(f"After detection frame {frame_count}")
                cv2.imwrite(f"temp/{self.job_id}/detection_output_{frame_count}.jpg", img)
            # If no person detected, use previous center(ceter of the frame)
            if center_x == -1 or center_y == -1:
                center_x, center_y = self.frame_width // 2, self.frame_height // 2
                
            self.center_x_list.append(center_x)
            self.center_y_list.append(center_y)
            pre_center_x, pre_center_y = center_x, center_y
            
            if frame_count % 100 == 0:
                logging.info(f"Processing frame: {frame_count}")
                
                
            progress = int((frame_count / self.total_frames) * 50) 
            self.job_status[self.job_id]["progress"] = min(progress, 50)
            frame_count += 1
        self.cap.release()
        
        # first_pass_metrics = self.profiler.stop()
        logging.info("\nFirst pass metrics:")
        # logging.info(f"Time taken: {first_pass_metrics['execution_time']:.2f} seconds")
        # logging.info(f"Memory used: {first_pass_metrics['memory_used']:.2f} MB")
        # logging.info(f"Peak memory: {first_pass_metrics['peak_memory']:.2f} MB")
        
        # self.profiler.start()
        
        # Smooth trajectories
        smoothed_x = TrajectorySmoothing.centered_moving_average(self.center_x_list)
        smoothed_y = TrajectorySmoothing.centered_moving_average(self.center_y_list)
        
        # Add Gaussian noise to the smoothed trajectories
        noise_std = 0.0001 * (self.frame_width + self.frame_height) / 2
        smoothed_x = np.array(smoothed_x) + np.random.normal(0, noise_std, len(smoothed_x))
        smoothed_y = np.array(smoothed_y) + np.random.normal(0, noise_std, len(smoothed_y))
        smoothed_x = np.clip(smoothed_x, 0, self.frame_width - 1)
        smoothed_y = np.clip(smoothed_y, 0, self.frame_height - 1)
        logging.info(f"Trajectory smoothing completed in {time.time() - start_time:.2f} seconds")
        logging.info(f"Total frames processed: {frame_count}")
        
        # Update job status
        #self.job_status[self.job_id]["progress"] = 10
        
        # float Nan check in w_margin_list and h_margin_list
        if not detector.w_margin_list or not detector.h_margin_list:
            logging.warning("No valid margins detected. Using default margins.")
            detector.w_margin_list = [self.frame_width // 4]
            detector.h_margin_list = [self.frame_height // 4]
        if np.isnan(detector.w_margin_list).any() or np.isnan(detector.h_margin_list).any():
            logging.warning("NaN values found in margins. Using default margins.")
            detector.w_margin_list = [self.frame_width // 4]
            detector.h_margin_list = [self.frame_height // 4]
        
        # Calculate margins
        w_margin = int(np.mean(detector.w_margin_list))
        h_margin = int(np.mean(detector.h_margin_list))
        
        # Ensure minimum margins
        if w_margin < self.frame_width//4 or h_margin < self.frame_height//4:
            w_margin = self.frame_width//4
            h_margin = self.frame_height//4
            
        if self.config.square_mode:
            margin = max(w_margin, h_margin)
            w_margin = h_margin = margin
            
        # Second pass: Crop and save frames
        self.cap = cv2.VideoCapture(self.input_path)
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            cropped_frame = self._crop_frame(
                frame, smoothed_x[frame_count], smoothed_y[frame_count], 
                w_margin, h_margin
            )
            self.frames.append(cropped_frame)
            
            if frame_count % 100 == 0:
                logging.info(f"Cropping frame: {frame_count}")
                # self.profiler.log_memory_usage(f"After cropping frame {frame_count}")
                
            progress = int((frame_count / self.total_frames) * 50) + 50
            self.job_status[self.job_id]["progress"] = min(progress, 80)
            frame_count += 1
            
        self.cap.release()
        
        # second_pass_metrics = self.profiler.stop()
        logging.info("\nSecond pass metrics:")
        # logging.info(f"Time taken: {second_pass_metrics['execution_time']:.2f} seconds")
        # logging.info(f"Memory used: {second_pass_metrics['memory_used']:.2f} MB")
        # logging.info(f"Peak memory: {second_pass_metrics['peak_memory']:.2f} MB")
        
        # Save video
        # self.profiler.start()
        self._save_video()

        # return True
        # save_metrics = self.profiler.stop()
        
        logging.info("\nVideo saving metrics:")
        # logging.info(f"Time taken: {save_metrics['execution_time']:.2f} seconds")
        # logging.info(f"Memory used: {save_metrics['memory_used']:.2f} MB")
        # logging.info(f"Peak memory: {save_metrics['peak_memory']:.2f} MB")
        
    def _save_video(self):

        self.job_status[self.job_id]["status"] = "converting"

        clip = ImageSequenceClip(self.frames, fps=self.fps)
        # Add audio
        try:
            audio = VideoFileClip(self.input_path).audio
            clip = clip.set_audio(audio)
        except Exception as e:
            logging.warning(f"Could not add audio: {str(e)}")
        
        self.job_status[self.job_id]["progress"] = 90
        
        clip.write_videofile(
            self.output_path, 
            codec='libx264',
            audio_codec='aac',
            verbose=False
        )
            
        # 작업 완료
        self.job_status[self.job_id]["status"] = "completed"
        self.job_status[self.job_id]["progress"] = 100
        self.job_status[self.job_id]["output_file"] = self.output_path
        logging.info(f"Video saved to: {self.output_path}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output video path')
    parser.add_argument('--version', '-v', type=int, default=8, help='YOLO version (5 or 8)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize configuration
    config = CropperConfig(model_version=args.version)
    
    # Create and setup video cropper
    cropper = VideoCropper(config)
    cropper.load_model()
    
    # Process video
    cropper.process_video(args.input, args.output)

# Initialize configuration
config = CropperConfig(model_version=11)
# Create and setup video cropper
cropper = VideoCropper(config)
cropper.load_model()


if __name__ == '__main__':
    main() 