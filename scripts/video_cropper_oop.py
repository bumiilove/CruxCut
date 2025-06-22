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
    # 비디오 샘플링 기본 프레임 간격 (기본 탐지 주기)
    sampling_rate_default: int = 30
    # 타이트 샘플링(더 촘촘한 탐지 주기, 필요시 사용)
    sampling_rate_tight: int = 5
    # 마진 비율(탐지된 바운딩 박스에 곱해 크롭 영역 확장)
    margin_ratio: float = 1.6
    # 정사각형 크롭 모드 여부(True면 정사각형으로 크롭)
    square_mode: bool = True
    # 포즈 추정 사용 여부(현재 미사용)
    do_pose_estimation: bool = False
    # 객체 탐지 결과 시각화 여부
    show_object_detection: bool = False
    # 탐지 이미지 저장 여부
    save_detection: bool = True
    # 사용할 YOLO 모델 버전(5, 8, 11 등)
    model_version: int = 11
    # 객체 탐지 신뢰도 임계값
    confidence_threshold: float = 0.3
    
class VideoCropper:
    """
    VideoCropper 클래스
    - 비디오 크롭 파이프라인 전체를 관리하는 클래스입니다.
    - 설정(config)과 모델, 탐지기, 프로세서를 관리합니다.
    """
    def __init__(self, config: Optional[CropperConfig] =  None):
        # 설정값 저장
        self.config = config or CropperConfig()
        self.model = None
        self.device = self._get_device()
        self.detector = None
        self.processor = None
        
    def _get_device(self) -> torch.device:
        # 사용 가능한 디바이스(CUDA/MPS/CPU) 자동 선택
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
    def load_model(self):
        """
        YOLO 모델을 설정값에 따라 불러오고, ObjectDetector를 초기화합니다.
        """
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
        """
        비디오 크롭 전체 파이프라인 실행 함수
        - 입력 비디오를 받아 탐지, 트래킹, 스무딩, 크롭, 저장까지 수행합니다.
        """
        self.processor = VideoProcessor(input_path, output_path, job_status, jobid, self.config)
        self.processor.process_video(self.detector)
        return True

class ObjectDetector:
    """
    ObjectDetector 클래스
    - YOLO 모델을 이용해 프레임에서 사람(클래스 0) 객체를 탐지합니다.
    - 탐지된 중심점 좌표와 마진 정보를 기록합니다.
    """
    def __init__(self, model: YOLO, config: CropperConfig):
        # YOLO 모델과 설정값 저장
        self.model = model
        self.config = config
        # 각 프레임별 마진(너비, 높이) 리스트
        self.w_margin_list: List[float] = []
        self.h_margin_list: List[float] = []
        
    def detect_person(self, frame: np.ndarray, prev_center_x: float, prev_center_y: float, frame_index: int) -> Tuple[float, float]:
        """
        한 프레임에서 사람 객체를 탐지하여 중심 좌표와 마진을 반환합니다.
        - 중심에 가까운 사람 1명을 선택합니다.
        - 탐지 실패 시 이전 중심 좌표를 반환합니다.
        - 탐지 결과 이미지를 저장할 수 있습니다.
        """
        # 프레임 너비 기반 탐지 가능한 중심 영역 정의
        frame_width = frame.shape[1]
        focus_zone_start = frame_width // 6
        focus_zone_end = frame_width * 5 // 6
        frame_center_x = frame_width // 2

        # YOLO 객체 탐지 실행
        results = self.model(frame, stream=True, verbose=False)
        selected_center_x, selected_center_y = -1, -1
        min_distance_to_center = float('inf')

        person_detected = False
        img = frame  # 기본값: 원본 프레임

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
    """
    TrajectorySmoothing 클래스
    - 중심점 좌표 리스트에 centered moving average(중앙 이동평균)로 스무딩 처리를 수행합니다.
    """
    @staticmethod
    def centered_moving_average(data: List[float], window_size: int = 10) -> List[float]:
        """
        중심 이동평균을 적용하여 좌표의 급격한 변화(진동 등)를 부드럽게 만듭니다.
        """
        avg_list = []
        for i in range(len(data)):
            start = max(0, i - window_size)
            end = min(len(data), i + window_size)
            avg = np.mean(data[start:end])
            avg_list.append(avg)
        return avg_list

class VideoProcessor:
    """
    VideoProcessor 클래스
    - 비디오의 프레임을 읽고, 탐지/스무딩/크롭/저장 전체 파이프라인을 담당합니다.
    - 프레임별 중심점 및 마진 계산, 트래킹, 최종 비디오 저장까지 수행합니다.
    """
    def __init__(self, input_path: str, output_path: str, job_status: dict, jobid: str, config: CropperConfig):
        # 비디오 입출력 경로, 설정, 상태 관리 등 저장
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
        """
        비디오 캡처 객체를 초기화하고, 프레임 해상도/프레임수/FPS 등 정보를 읽어옵니다.
        """
        self.cap = cv2.VideoCapture(self.input_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 작업 상태 초기화
        self.job_status[self.job_id] = {
            "status": "processing",
            "progress": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": self.input_path,
            "output_file": self.output_path,
        }

    def _crop_frame(self, frame: np.ndarray, center_x: float, center_y: float, 
                   w_margin: float, h_margin: float) -> np.ndarray:
        """
        입력 프레임에서 중심점과 마진을 기준으로 크롭을 수행하고, 크롭 영역이 부족하면 블랭크 패딩을 추가합니다.
        """
        frame_height, frame_width = frame.shape[:2]
        
        crop_x1 = int(max(center_x - w_margin, 0))
        crop_y1 = int(max(center_y - h_margin, 0))
        crop_x2 = int(min(center_x + w_margin, frame_width))
        crop_y2 = int(min(center_y + h_margin, frame_height))
        
        # 크롭 영역이 마진보다 작으면 보정
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
        
        # 블랭크(검정색) 프레임 생성 후, 크롭 영역을 정중앙에 삽입
        blank_frame = np.zeros((int(h_margin * 2), int(w_margin * 2), 3), dtype=np.uint8)
        cropped_part = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_h, cropped_w = cropped_part.shape[:2]
        
        start_y = (blank_frame.shape[0] - cropped_h) // 2
        start_x = (blank_frame.shape[1] - cropped_w) // 2
        blank_frame[start_y:start_y + cropped_h, start_x:start_x + cropped_w] = cropped_part
        
        return cv2.cvtColor(blank_frame, cv2.COLOR_RGB2BGR)
    
    def process_video(self, detector: ObjectDetector):
        """
        비디오 전체 처리 파이프라인
        1. 프레임별 사람 탐지 및 중심점 좌표 수집
        2. 중심점 트래젝토리 스무딩
        3. 마진 계산
        4. 프레임 크롭
        5. 최종 비디오 저장
        """
        self._init_video_capture()
        logging.info(f"Processing video: {self.input_path}")
        logging.info(f"Frame size: {self.frame_width}x{self.frame_height}, FPS: {self.fps}")
        
        os.makedirs(f"temp/{self.job_id}", exist_ok=True)
        
        pre_center_x, pre_center_y = -1, -1
        frame_count = 0
        start_time = time.time()
        
        # 1. 첫 번째 패스: 탐지 및 중심점 좌표 수집
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 초반 프레임은 탐지 간격을 짧게 설정, 이후에는 기본 탐지 간격(sampling_rate_default) 사용
            if  frame_count < 90 or frame_count % self.config.sampling_rate_default == 0:
                center_x, center_y, img = detector.detect_person(frame, pre_center_x, pre_center_y, frame_count)
                # 탐지 결과 이미지 저장
                cv2.imwrite(f"temp/{self.job_id}/detection_output_{frame_count}.jpg", img)
            # 탐지 실패 시 프레임 중앙 사용
            if center_x == -1 or center_y == -1:
                center_x, center_y = self.frame_width // 2, self.frame_height // 2
                
            self.center_x_list.append(center_x)
            self.center_y_list.append(center_y)
            pre_center_x, pre_center_y = center_x, center_y
            
            if frame_count % 100 == 0:
                logging.info(f"Processing frame: {frame_count}")
                
            # 진행률 갱신(1차 패스: 0~50%)
            progress = int((frame_count / self.total_frames) * 50) 
            self.job_status[self.job_id]["progress"] = min(progress, 50)
            frame_count += 1
        self.cap.release()
        
        logging.info("\nFirst pass metrics:")
        
        # 2. 중심점 트래젝토리 스무딩
        smoothed_x = TrajectorySmoothing.centered_moving_average(self.center_x_list)
        smoothed_y = TrajectorySmoothing.centered_moving_average(self.center_y_list)
        
        # 3. 스무딩 좌표에 가우시안 노이즈 추가(더 부드럽게)
        noise_std = 0.0001 * (self.frame_width + self.frame_height) / 2
        smoothed_x = np.array(smoothed_x) + np.random.normal(0, noise_std, len(smoothed_x))
        smoothed_y = np.array(smoothed_y) + np.random.normal(0, noise_std, len(smoothed_y))
        smoothed_x = np.clip(smoothed_x, 0, self.frame_width - 1)
        smoothed_y = np.clip(smoothed_y, 0, self.frame_height - 1)
        logging.info(f"Trajectory smoothing completed in {time.time() - start_time:.2f} seconds")
        logging.info(f"Total frames processed: {frame_count}")
        
        # 4. 마진 계산(탐지 결과 기반, 없으면 기본값 사용)
        if not detector.w_margin_list or not detector.h_margin_list:
            logging.warning("No valid margins detected. Using default margins.")
            detector.w_margin_list = [self.frame_width // 4]
            detector.h_margin_list = [self.frame_height // 4]
        if np.isnan(detector.w_margin_list).any() or np.isnan(detector.h_margin_list).any():
            logging.warning("NaN values found in margins. Using default margins.")
            detector.w_margin_list = [self.frame_width // 4]
            detector.h_margin_list = [self.frame_height // 4]
        
        w_margin = int(np.mean(detector.w_margin_list))
        h_margin = int(np.mean(detector.h_margin_list))
        
        # 최소 마진 보장
        if w_margin < self.frame_width//4 or h_margin < self.frame_height//4:
            w_margin = self.frame_width//4
            h_margin = self.frame_height//4
            
        # 정사각형 모드면 마진 통일
        if self.config.square_mode:
            margin = max(w_margin, h_margin)
            w_margin = h_margin = margin
            
        # 5. 두 번째 패스: 스무딩된 좌표로 프레임 크롭 및 저장
        self.cap = cv2.VideoCapture(self.input_path)
        frame_count = 0
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
                
            # 진행률 갱신(2차 패스: 50~80%)
            progress = int((frame_count / self.total_frames) * 50) + 50
            self.job_status[self.job_id]["progress"] = min(progress, 80)
            frame_count += 1
            
        self.cap.release()
        
        logging.info("\nSecond pass metrics:")
        
        # 6. 최종 비디오 저장
        self._save_video()

        logging.info("\nVideo saving metrics:")
        
    def _save_video(self):
        """
        크롭된 프레임 시퀀스를 비디오 파일로 저장하고, 오디오도 함께 붙입니다.
        """
        self.job_status[self.job_id]["status"] = "converting"

        clip = ImageSequenceClip(self.frames, fps=self.fps)
        # 오디오 추가
        try:
            audio = VideoFileClip(self.input_path).audio
            clip = clip.set_audio(audio)
        except Exception as e:
            logging.warning(f"Could not add audio: {str(e)}")
        
        clip.write_videofile(
            self.output_path, 
            codec='libx264',
            audio_codec='aac',
            verbose=False
        )
            
        # 작업 완료 상태 갱신
        self.job_status[self.job_id]["status"] = "completed"
        self.job_status[self.job_id]["progress"] = 100
        self.job_status[self.job_id]["output_file"] = self.output_path
        logging.info(f"Video saved to: {self.output_path}")

def main():
    """
    main 함수
    - 커맨드라인 인자 파싱 후 파이프라인 전체 실행
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input video path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output video path')
    parser.add_argument('--version', '-v', type=int, default=8, help='YOLO version (5 or 8)')
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 설정 초기화
    config = CropperConfig(model_version=args.version)
    
    # VideoCropper 생성 및 모델 로드
    cropper = VideoCropper(config)
    cropper.load_model()
    
    # 비디오 처리 실행
    cropper.process_video(args.input, args.output)

# Global 변수로 CropperConfig와 VideoCropper 인스턴스 생성
# Initialize configuration
config = CropperConfig(model_version=11)
# Create and setup video cropper
cropper = VideoCropper(config)
cropper.load_model()


if __name__ == '__main__':
    main() 