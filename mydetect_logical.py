import sys
import cv2
import time
import argparse
import torch
import os

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import threading
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse


import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                           check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
# is_detecting: bool = True
# Parameters
HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from mydetect_ui import Ui_Form  # 导入detect_ui的界面


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class UI_Logic_Window(QtWidgets.QWidget, LoadStreams):
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        self.timer_video = QtCore.QTimer()  # 创建定时器
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.init_slots()
        # self.cap = cv2.VideoCapture()
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        # self.output_folder = 'output/'
        # self.vid_writer = None

        # 权重初始文件名
        # self.openfile_name_model = None
    def init_slots(self):
        self.ui.pushButton.clicked.connect(self.onButtonClick)
        # self.ui.pushButton_img.clicked.connect(self.button_image_open)
        # self.ui.pushButton_video.clicked.connect(self.button_video_open)
        # self.ui.pushButton_camer.clicked.connect(self.button_camera_open)
        # self.ui.pushButton_weights.clicked.connect(self.open_model)
        # self.ui.pushButton_init.clicked.connect(self.model_init)
        # self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        # self.ui.pushButton_finish.clicked.connect(self.finish_detect)

    def model_init(self, index0, index1, index2, index3):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default= 'weights/yolov5s.pt',help='model path or triton URL')
        parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')

        if int(index0) == 1 and int(index1) == int(index2) == int(index3) == 0:  # 1000
            parser.add_argument('--classes', nargs='+', type=int, default=0,
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index0) == int(index1) == 1 and int(index2) == int(index3) == 0:  # 1100
            parser.add_argument('--classes', nargs='+', type=int, default=[0, 1],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index0) == int(index1) == int(index2) == 1 and int(index3) == 0:  # 1110
            parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index1) == 1 and int(index0) == int(index2) == int(index3) == 0:  # 0100
            parser.add_argument('--classes', nargs='+', type=int, default=1,
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index2) == 1 and int(index0) == int(index1) == int(index3) == 0:  # 0010
            parser.add_argument('--classes', nargs='+', type=int, default=2,
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index3) == 1 and int(index0) == int(index1) == int(index2) == 0:  # 0001
            parser.add_argument('--classes', nargs='+', type=int, default=3,
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index1) == int(index2) == 1 and int(index0) == int(index3) == 0:  # 0110
            parser.add_argument('--classes', nargs='+', type=int, default=[1, 2],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index2) == int(index3) == 1 and int(index0) == int(index1) == 0:  # 0011
            parser.add_argument('--classes', nargs='+', type=int, default=[2, 3],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index0) == int(index2) == 1 and int(index1) == int(index3) == 0:  # 1010
            parser.add_argument('--classes', nargs='+', type=int, default=[0, 2],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index0) == int(index3) == 1 and int(index1) == int(index2) == 0:  # 1001
            parser.add_argument('--classes', nargs='+', type=int, default=[0, 3],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index1) == int(index3) == 1 and int(index0) == int(index2) == 0:  # 0101
            parser.add_argument('--classes', nargs='+', type=int, default=[1, 3],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index1) == int(index2) == int(index3) == 1 and int(index0) == 0:  # 0111
            parser.add_argument('--classes', nargs='+', type=int, default=[1, 2, 3],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index0) == int(index2) == int(index3) == 1 and int(index1) == 0:  # 1011
            parser.add_argument('--classes', nargs='+', type=int, default=[0, 2, 3],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        elif int(index0) == int(index1) == int(index3) == 1 and int(index2) == 0:  # 1101
            parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 3],
                                help='filter by class: --classes 0, or --classes 0 2 3')
        else:  # 0000 1111
            parser.add_argument('--classes', nargs='+', type=int,
                                help='filter by class: --classes 0, or --classes 0 2 3')

        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print(vars(opt))
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        self.source = str(source)
        self.save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        self.is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = source.isnumeric() or source.endswith('.txt') or (self.is_url and not self.is_file)
        # screenshot = source.lower().startswith('screen')
        if self.is_url and self.is_file:
            source = check_file(self.source)  # download

        self.save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok)  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        # Load model
        self.device = select_device(self.opt.device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.opt.dnn, data=self.opt.data, fp16=self.opt.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def detect(self):
        self.bs = 1
        if self.webcam:
            self.view_img = check_imshow(warn=True)
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
            self.bs = len(dataset)
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = self.opt.model(im, augment=self.opt.augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=self.opt.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.opt.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.opt.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.opt.save_img or self.opt.save_crop or self.opt.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.opt.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if self.opt.view_img:
                    # 选择字体和大小
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2

                    # 选择文本的位置
                    text_position = (50, 50)  # (x, y) 坐标，从左上角开始

                    # 选择文本的颜色和粗细
                    text_color = (255, 0, 0)  # BGR 格式，这里是红色
                    text_thickness = 2

                    # if platform.system() == 'Linux' and p not in windows:
                    # windows.append(p)  # 创建窗口，允许调整大小
                    if str(p) == "0":
                        label_text = "0"
                        im0_resized = cv2.resize(im0, (960, 1080))
                        # 在图像上添加文本
                        cv2.putText(im0_resized, label_text, text_position, font, font_scale, text_color,
                                    font_thickness)

                    elif str(p) == "1":
                        label_text = "1"
                        im1_resized = cv2.resize(im0, (960, 1080))
                        # 在图像上添加文本
                        cv2.putText(im1_resized, label_text, text_position, font, font_scale, text_color,
                                    font_thickness)
                        combined_frame = cv2.hconcat([im0_resized, im1_resized])
                        cv2.imshow("dd", combined_frame)
                        cv2.waitKey(1)  # 1 millisecond
                        # return combined_frame

    def getCheckboxStates(self):
        # 获取四个checkbox按钮的状态
        state0 = 1 if self.checkBox.isChecked() else 0
        state1 = 1 if self.checkBox_2.isChecked() else 0
        state2 = 1 if self.checkBox_3.isChecked() else 0
        state3 = 1 if self.checkBox_4.isChecked() else 0
        listState = [state0, state1, state2, state3]
        print(listState)
        return listState

    def onButtonClick(self):

        checkbox_states = self.getCheckboxStates()
        self.model_init(checkbox_states[0], checkbox_states[1],checkbox_states[2], checkbox_states[3])
        self.detect()


class LoadStreams(UI_Logic_Window):
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`

    def __init__(self, sources='0', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):

        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for self.i, self.s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{self.i + 1}/{n}: {self.s}... '
            if urlparse(self.s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                self.s = pafy.new(self.s).getbest(preftype="mp4").url  # YouTube URL
            self.s = eval(self.s) if self.s.isnumeric() else self.s  # i.e. s = '0' local webcam
            if self.s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            self.cap = int(cv2.VideoCapture(self.s))
            assert self.cap.isOpened(), f'{st}Failed to open {self.s}'
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH, ))
            print(w)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT, ))
            print(h)
            fps = self.cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[self.i] = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[self.i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[self.i] = self.cap.read()  # guarantee first frame
            self.threads[self.i] = Thread(target=self.update, args=([self.i, self.cap, self.s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[self.i]} frames {w}x{h} at {self.fps[self.i]:.2f} FPS)")
            self. threads[self.i].start()

        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):

        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1

            cap.grab()  # .read() = .grab() followed by .retrieve()
            # while not self.state:  # 当标志变量为 False 时等待
            #     time.sleep(0.1)
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''
    # def pause_detection(self):
    #     LoadStreams.state = False
    # def continue_detection(self):
    #     LoadStreams.state = True

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())