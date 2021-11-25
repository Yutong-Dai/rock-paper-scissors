# -*-coding:utf-8-*-
# date:2021-03-16
# Author: Eric.Lee
# function: yolo v5 video inference

import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
sys.path.append("../")
from src.yolohand.utils.datasets import *
import src.yolohand.utils.torch_utils as torch_utils
sys.path.insert(0, "../src/yolohand")
from src.yolohand.utils.general import non_max_suppression, scale_coords
import time


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect(weights='yolohand/runs/train/exp/weights/best.pt', half=False, imgsz=640, device='cpu',
           conf_thres=0.3, iou_thres=0.5):
    # setup
    device = torch_utils.select_device(device)
    model = torch.load(weights, map_location=device)['model']
    if device.type == 'cpu':
        model.to(device).float().eval()
    else:
        model.to(device).eval()
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # ============================== Run inference =====================
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()
              ) if device.type != 'cpu' else None  # run once

    vid_cap = cv2.VideoCapture(0)

    # calculate fps
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        ret, img0 = vid_cap.read()
        if ret:
            # image resize
            img = letterbox(img0, new_shape=imgsz)[0]
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # prediction
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=False)[0]
            t2 = torch_utils.time_synchronized()

            if half:
                pred = pred.float()

            # only one anchor is responsible for prediction
            pred = non_max_suppression(pred, conf_thres, iou_thres,
                                       classes=None, agnostic=False)
            # detections per image
            for _, det in enumerate(pred):

                s, im0 = '', img0

                s += '%gx%g ' % img.shape[2:]  # print string
                #  normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # Write results
                    output_dict_ = []
                    for *xyxy, conf, cls in det:

                        x1, y1, x2, y2 = xyxy
                        output_dict_.append(
                            (float(x1), float(y1), float(x2), float(y2)))
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=3)

                print('%sDone. (%.3fs)' % (s, t2 - t1))
            # calculate fps
            new_frame_time = time.time()
            fps = str(int(1 / (new_frame_time - prev_frame_time)))
            prev_frame_time = new_frame_time
            cv2.putText(im0, f"Paper-Rock-Scissors:{fps} FPS", (5, im0.shape[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (100, 255, 0), 3, cv2.LINE_AA)

            cv2.namedWindow("RPS GAME", 0)
            cv2.imshow("RPS GAME", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        detect()
