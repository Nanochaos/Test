import argparse
import os
from pathlib import Path

import numpy
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import json, zlib, base64


# Reduce memory usage and speeds up computation by disabling gradient calculation
@torch.no_grad()
class Detection:
    def __init__(self, weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt,
                 save_conf, save_crop,
                 nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness,
                 hide_labels,
                 hide_conf, half, dnn, save_results):
        self.weights = weights  # model.pt path(s)
        self.source = source  # file/dir/URL/glob, 0 for webcam
        self.data = data  # dataset.yaml path
        self.imgsz = imgsz  # inference size (height, width)
        # Classifies an object depending on the confidence threshold
        self.conf_thres = conf_thres  # confidence threshold
        # Threshold for removing overlapping bounding boxes
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = max_det  # maximum detections per image
        self.device = device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = view_img  # show results
        self.save_txt = save_txt  # save results to *.txt
        self.save_conf = save_conf  # save confidences in --save-txt labels
        self.save_crop = save_crop  # save cropped prediction boxes
        self.nosave = nosave  # do not save images/videos
        # Class 0 is head detections; Class 1 is visible body detections
        self.classes = classes  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = agnostic_nms  # class-agnostic NMS
        self.augment = augment  # augmented inference
        self.visualize = visualize  # visualize features
        self.update = update  # update all models
        self.project = project  # save results to project/name
        self.name = name  # save results to project/name
        self.exist_ok = exist_ok  # existing project/name ok, do not increment
        self.line_thickness = line_thickness  # bounding box thickness (pixels)
        self.hide_labels = hide_labels  # hide labels
        self.hide_conf = hide_conf  # hide confidences
        self.half = half  # use FP16 half-precision inference
        self.dnn = dnn  # use OpenCV DNN for ONNX inference

        self.save_results = save_results
        self.results = {}

    def run(self):
        self.source = str(self.source)
        save_img = not self.nosave and not self.source.endswith('.txt')  # save inference LoadImages
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)  # download

        # Directories
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        if not self.nosave:
            (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make directory

        # Load model
        self.device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            self.visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            pred = model(im, augment=self.augment, visualize=self.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (
                                names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    # Save detections in matrix format
                    if self.save_results:
                        m = np.zeros(im0s.shape[:2])
                        for e in det[:, :4]:
                            x1, y1, x2, y2 = e
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            m[y1:y2, x1:x2] += 1

                        # Save matrix
                        if not hasattr(dataset, 'frame'):
                            dataset.frame = 1
                        self.results[dataset.frame] = {'detections': len(det), 'matrix': m.tolist()}
                        os.mkdir(save_dir / 'results') if not os.path.exists(save_dir / 'results') else None

                        # save to compressed numpy file
                        with open(save_dir / 'results' / f'{p.stem}.compressed', 'w') as f:
                            f.write(
                                base64.b64encode(
                                    zlib.compress(
                                        json.dumps(self.results).encode('utf-8')
                                    )
                                ).decode('ascii')
                            )

                        # decode compressed numpy file
                        # with open(save_dir / 'results' / f'{p.stem}.compressed', 'r') as f:
                        #     self.results = json.loads(zlib.decompress(base64.b64decode(f.read())))

                        # if dataset.frame == 20:
                        #     raise Exception('stop')

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(
                                    Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)

            """ REMOVE THIS ONCE YOU CAN STREAM THE IMAGES IN THE GUI"""
            # Stream results
            im0 = annotator.result()
            if self.view_img and im0.any():
                # cv2.imshow(str(p), cv2.resize(im0, (1920, 1080)))
                # cv2.imshow(str(p), im0)
                cv2.imshow(f'Frame: {dataset.frame} File:{str(p)}', cv2.resize(im0, (round(im0.shape[1] * 0.5),
                                                                                     round(im0.shape[0] * 0.5))))
                if webcam:
                    cv2.waitKey(1)  # 1 millisecond
                else:
                    cv2.waitKey(1)

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)
        cv2.waitKey(0)


def main():
    # We change variables here
    Test = Detection(**vars(opt))
    Test.source = '5.mp4'
    Test.conf_thres = 0.2
    Test.iou_thres = 0.3
    Test.hide_labels = True
    Test.line_thickness = 2
    Test.classes = 0
    Test.save_txt = True
    Test.save_img = True
    Test.save_crop = True
    Test.save_conf = True
    Test.nosave = False
    Test.save_results = True
    Test.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='crowdhuman_vbody_yolov5m.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(640, 640),
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='show results')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=False, action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', default=0, nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize features')
    parser.add_argument('--update', default=False, action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=True, action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')

    parser.add_argument('--save-results', default=False, action='store_true', help='save matrix')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    check_requirements(exclude=('tensorboard', 'thop'))
    main()
