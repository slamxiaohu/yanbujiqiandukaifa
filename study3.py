# 开发人员：刘奇龙
# 开发日期：2024.1.28

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QMenu, QAction
from studyui import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import argparse
import os
import json
import platform
import sys
from pathlib import Path

import torch
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPoint
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np

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


class DetThread(QThread):
    send_img1 = pyqtSignal(np.ndarray)  # 用于发送检测结果的图像,4个检测结果图像
    send_img2 = pyqtSignal(np.ndarray)  # 用于发送检测结果的图像,4个检测结果图像
    send_img3 = pyqtSignal(np.ndarray)  # 用于发送检测结果的图像,4个检测结果图像
    send_img4 = pyqtSignal(np.ndarray)  # 用于发送检测结果的图像,4个检测结果图像
    send_statistic = pyqtSignal(dict)  # 用于目标检测的统计信息，通常包括关于检测到的目标数量、类别等方面信息数据
    send_msg = pyqtSignal(str)  # 发送消息，可以是关于检测状态的信息
    # send_fps = pyqtSignal(str)  # 发送每秒处理的帧信息

    # 初始化方法
    def __init__(self):
        super(DetThread, self).__init__()
        # 初始化属性信息
        self.weights = ROOT/'yolov5s.pt'
        self.source = 0
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = []
        # 初始化控制检测流程的标志
        self.jump_out = False  # 设置是否跳出循环
        self.is_continue = True  # 设置是否继续进行检测
        self.save_fold = './result'  # 保存检测结果的文件夹路径
        self.min_box_width = 0.0
        self.max_box_width = 100.0
        self.min_box_height = 0.0
        self.max_box_height = 100.0


    @smart_inference_mode()
    def run(self,
            # classes = None,
            source= 0,
            imgsz=(640, 640),  # inference size (height, width)
            max_det=1000,  # maximum detections per image
            device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        try:
            source = str(self.source)
            webcam = source.isnumeric() or source.endswith('.txt')

            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(self.weights, device=device, dnn=dnn, data=None, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Dataloader
            bs = 1  # batch_size
            if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

            dataset = iter(dataset)
            print(dataset)
            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):  # 检查是否存在属性 self.out。这可能是检查是否有视频写入器对象。
                        self.out.release()  # 如果存在视频写入器对象 self.out，则释放它。这是为了停止将处理后的帧写入输出视频。
                    break

                if self.is_continue:
                    path, im, im0s, self.vid_cap, *_ = next(dataset)

                    statistic_dic = {name: 0 for name in names}
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    # Inference
                    pred = model(im, augment=augment, visualize=visualize)
                    # NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, agnostic_nms,
                                               max_det=max_det)
                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        # im0 = im0s.copy()
                        # im0 = torch.tensor(im0) if isinstance(im0, list) else im0
                        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        try:
                            print("Before Annotator initialization")
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                            print("After Annotator initialization")
                        except Exception as e:
                            print(f"An error occurred: {e}")

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()  # detections per class
                                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            for *xyxy, conf, cls in reversed(det):

                                c = int(cls)  # integer class
                                print("开始")
                                box_width = (xyxy[2] - xyxy[0]).item()  # x_max - x_min
                                box_height = (xyxy[3] - xyxy[1]).item() # y_max - y_min
                                print(box_height, box_width)
                                print("结束")

                                if (self.min_box_width <= box_width <= self.max_box_width) and (self.min_box_height <= box_height <= self.max_box_height):
                                    try:
                                        print("Before conditional expression")
                                        if names[c] not in statistic_dic:
                                            statistic_dic[names[c]] = 0
                                        statistic_dic[names[c]] += 1

                                        # statistic_dic[names[c]] += 1
                                        print("After conditional expression")
                                        label = None if hide_labels else (
                                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                        annotator.box_label(xyxy, label, color=colors(c, True))
                                        print(xyxy)
                                        print("After Annotator initialization")
                                    except Exception as e:
                                        print(f"An error occurred: {e}")

                    # Stream results
                        im0 = annotator.result()
                        if str(p) == '0':
                            self.send_img1.emit(im0)

                        elif str(p) == '1':
                            self.send_img2.emit(im0)
                    # elif str(p) == '2':
                    #     self.send_img3.emit(im0)
                    # else:
                    #     self.send_img4.emit(im0)

        except Exception as e:
            self.send_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)

        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)
        # self.classes = self.getCheckboxStates()
        self.det_thread = DetThread()

        self.det_thread.weights = ROOT / 'yolov5s.pt'
        self.det_thread.source = 'streams.txt'
        # self.det_thread.classes = self.getCheckboxStates()

        self.det_thread.send_img1.connect(lambda x: self.show_image(x, self.out_video1))
        self.det_thread.send_img2.connect(lambda x: self.show_image(x, self.out_video2))
        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        # self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        # self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))

        self.load_setting()


    def getCheckboxStates(self):
        # 获取四个checkbox按钮的状态
        state0 = 1 if self.checkBox.isChecked() else 0
        state1 = 1 if self.checkBox_2.isChecked() else 0
        state2 = 1 if self.checkBox_3.isChecked() else 0
        state3 = 1 if self.checkBox_4.isChecked() else 0
        listState = [state0, state1, state2, state3]
        indexes_of_ones = [index for index, value in enumerate(listState) if value == 1]
        print(indexes_of_ones)
        return indexes_of_ones

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.det_thread.conf_thres = x
        elif flag == 'iouSpinBox':
            self.det_thread.iou_thres = x
        else:
            pass

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.classes = self.getCheckboxStates()
            # self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        # self.saveCheckBox.setEnabled(True)

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)

    def load_setting(self):
        config_file = 'config/setting.json'
        config_dir = os.path.dirname(config_file)

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            # rate = 10
            # check = 0
            # savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf
                          # "rate": rate,
                          # "check": check,
                          # "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                # rate = 10
                # check = 0
                # savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                # rate = config['rate']
                # check = config['check']
                # savecheck = config['savecheck']
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        # self.checkBox.setCheckState(check)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())

