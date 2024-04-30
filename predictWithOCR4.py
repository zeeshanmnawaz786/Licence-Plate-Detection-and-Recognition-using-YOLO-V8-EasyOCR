# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(result[1]) > 6 and result[2] > conf:
            ocr = result[1]

    # Trim whitespace from OCR result and remove spaces
    ocr = ocr.strip().replace(" ", "")

    return str(ocr)

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tracker = cv2.TrackerKCF_create()
        self.plates_file = open('all_plates_data.txt', 'a')  # Open the file in append mode
        self.detected_plates = {}

    def __del__(self):
        if hasattr(self, 'plates_file'):
            self.plates_file.close()

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds
    
    def track_vehicle(self, frame, bbox):
        success, bbox = self.tracker.update(frame)
        if success:
            return bbox  # Updated bounding box
        else:
            return None  # Vehicle lost
    
    def get_speed(self, bbox_prev, bbox_curr, time_interval):
        # Calculate the displacement between the centroids of the two bounding boxes
        # Assuming time_interval is in seconds
        displacement = ((bbox_curr[0] + bbox_curr[2] / 2) - (bbox_prev[0] + bbox_prev[2] / 2))
        # Assuming pixel to meters conversion factor (you need to calibrate this)
        pixel_to_meters = 0.1  # Example value
        displacement_meters = displacement * pixel_to_meters
        # Speed = displacement / time
        speed = displacement_meters / time_interval
        return speed

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        bbox_prev = None
        time_interval = 1
        log_string = ""

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        speed = None  # Initialize speed variable
   
        for *xyxy, conf, cls in reversed(det):
            if bbox_prev is not None:
                bbox_curr = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                ocr = getOCR(im0, xyxy)
                if ocr and ocr in ["5723HSD", "2193BJB", "0791DVL", "6061GEC", "6896FMP", "7207DHR"]:
                    speed = self.get_speed(bbox_prev, bbox_curr, time_interval)
                    speed = abs(speed) if speed is not None and speed < 0 else speed
                    print(f"Vehicle with plate {ocr}: {speed} km/h")

                    # Check if the plate is already detected
                    if ocr in self.detected_plates:
                        # Replace the old speed with the new one
                        self.detected_plates[ocr] = speed
                        print(f"Updated speed for plate {ocr}: {speed} km/h")
                        
                        # Open the file in read mode to read existing content
                        with open('all_plates_data.txt', 'r') as file:
                            data = file.readlines()
                        
                        # Open the file again in write mode to overwrite the existing content
                        with open('all_plates_data.txt', 'w') as file:
                            for line in data:
                                # Update the speed entry for the plate
                                if f"Plate: {ocr}" in line:
                                    file.write(f"Plate: {ocr}, Speed: {speed:.2f} km/h\n")
                                else:
                                    file.write(line)
                    else:
                        # Plate is not detected before, add it to the dictionary
                        self.detected_plates[ocr] = speed
                        print(f"Added new plate {ocr} with speed: {speed} km/h")

                        # Write plate and speed information to .txt file
                        
                        self.plates_file.write(f"Plate: {ocr}, Speed: {speed:.2f} km/h\n")
                        self.plates_file.flush()  # Ensure data is written to file immediately

            bbox_prev = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]



            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0,xyxy)
                if ocr != "":
                    print("Detected Number Plate:", ocr)  # Print the detected number plate
                    label = ocr
                # Add speed annotation
                label += f", Speed: {speed:.2f} km/h" if speed is not None else ""
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                            imc,
                            file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                            BGR=True)

        return log_string



@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    print("predictor",predictor)
    predictor()


if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()
