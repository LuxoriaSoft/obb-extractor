import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image


def _expand_box(box, margin, max_width, max_height):
    x, y, w, h = box
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, max_width - 1)
    y2 = min(y + h + margin, max_height - 1)
    return (x1, y1, x2 - x1, y2 - y1)


def extract_bounding_boxes_threshold(image, threshold=127, invert=False, margin=0):
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {image}")
    else:
        img = image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold, 255, thresh_type)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    if margin > 0:
        h, w = img.shape[:2]
        boxes = [_expand_box(b, margin, w, h) for b in boxes]
    return boxes


def extract_bounding_boxes_model(image, confidence=0.5, device=None, local_weights_path=None, margin=0):
    device = torch.device(device if device else 'cpu')
    # load model
    if local_weights_path:
        model = fasterrcnn_resnet50_fpn(weights=None).to(device)
        state = torch.load(local_weights_path, map_location=device)
        model.load_state_dict(state)
    else:
        weights_enum = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights_enum).to(device)
    model.eval()

    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {image}")
    else:
        img = image.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = transforms.ToTensor()(pil_img).to(device)

    with torch.no_grad():
        outputs = model([tensor])

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    mask = scores >= confidence
    raw = boxes[mask]

    h, w = img.shape[:2]
    bboxes = []
    for x1, y1, x2, y2 in raw:
        box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        if margin > 0:
            box = _expand_box(box, margin, w, h)
        bboxes.append(box)
    return bboxes


def draw_bounding_boxes(image, bboxes, color=(0,255,0), thickness=2, margin=0):
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"image not found : {image}")
    else:
        img = image.copy()

    h, w = img.shape[:2]
    for box in bboxes:
        if margin > 0:
            x, y, bw, bh = box
            box = _expand_box(box, margin, w, h)
        x, y, bw, bh = box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, thickness)
    return img

if __name__ == '__main__':
    input_path = 'img.jpg'

    thr = extract_bounding_boxes_threshold(input_path, threshold=100, margin=10)
    print(f"Threshold  {len(thr)} boxes")

    m = extract_bounding_boxes_model(input_path, confidence=0.7, margin=5)
    print(f"Model : {len(m)} boxes")

    # save with additional 5px margin on draw
    out1 = draw_bounding_boxes(input_path, thr, margin=5)
    cv2.imwrite('out_thr.jpg', out1)
    out2 = draw_bounding_boxes(input_path, m, margin=5)
    cv2.imwrite('out_model.jpg', out2)
    print("Saved output images.")
